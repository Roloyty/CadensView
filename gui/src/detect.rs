//! 切镜检测：spawn `python -m shotlab.onnx_worker`（既有子进程协议）。
//! stdout JSON-lines: loading/device/progress/partial/complete/cancelled/error；
//! 取消 = 在 index JSON 同目录创建 cancel 文件（worker 侧协议），
//! 退出码 3 = GPU 错误，GUI 可在纯 CPU 新进程重试一次。
//!
//! cancel 文件的生命周期必须严格管理：spawn 前清残留、结束后删除、
//! Drop 时兜底删除，否则一次取消会永久污染后续所有检测。

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

/// worker 约定的取消哨兵路径：`--index` 参数的父目录下的 `cancel` 文件。
pub fn cancel_path(index_json: &Path) -> PathBuf {
    index_json
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("cancel")
}

/// 清理上一轮可能残留的取消文件；返回本轮使用的取消路径。
fn prepare_cancel(index_json: &Path) -> PathBuf {
    let path = cancel_path(index_json);
    let _ = std::fs::remove_file(&path);
    path
}

const STDERR_TAIL_LINES: usize = 8;

pub struct Detection {
    pub cancel_file: PathBuf,
    child: Child,
    /// worker stderr 的最后几行（traceback 结尾）。
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    /// stderr 读到 EOF 时发信号；进程退出后等它一小会儿，尾行才拿得全。
    stderr_done: Receiver<()>,
    reaped: bool,
}

impl Detection {
    pub fn spawn(
        python: &Path,
        repo_root: &Path,
        index_json: &Path,
        out_dir: &Path,
        weights: &Path,
        force_cpu: bool,
        on_event: impl Fn(Value) + Send + 'static,
    ) -> std::io::Result<Detection> {
        // 不能预创建 out_dir：worker 用独占语义写结果目录（exist_ok=False），
        // 目录已存在会让检测在 save_result 阶段报 WinError 183。
        let mut command = Command::new(python);
        command
            .current_dir(repo_root)
            .args(["-m", "shotlab.onnx_worker"])
            .arg("--index").arg(index_json)
            .arg("--out").arg(out_dir)
            .arg("--weights").arg(weights)
            .arg("--device").arg(if force_cpu { "cpu" } else { "auto" })
            .env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        Self::spawn_command(command, index_json, on_event)
    }

    fn spawn_command(
        mut command: Command,
        index_json: &Path,
        on_event: impl Fn(Value) + Send + 'static,
    ) -> std::io::Result<Detection> {
        let cancel_file = prepare_cancel(index_json);
        let mut child = command.spawn()?;
        let stdout = child.stdout.take().expect("worker stdout");
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                let Ok(value) = serde_json::from_str::<Value>(&line) else { continue };
                on_event(value);
            }
        });
        // Python 的 traceback 只走 stderr。worker 在自己 emit 之外崩溃时（导入失败、
        // 解释器本身出错），这几行是唯一能说明原因的东西，不能丢给 null。
        let stderr = child.stderr.take().expect("worker stderr");
        let stderr_tail = Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_TAIL_LINES)));
        let (done_tx, stderr_done) = channel();
        let tail = Arc::clone(&stderr_tail);
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let mut tail = tail.lock().unwrap();
                if tail.len() == STDERR_TAIL_LINES {
                    tail.pop_front();
                }
                tail.push_back(line.to_string());
            }
            let _ = done_tx.send(());
        });
        Ok(Detection { cancel_file, child, stderr_tail, stderr_done, reaped: false })
    }

    /// 请求取消：直接落地哨兵文件，worker 每 50ms 检查一次。
    pub fn cancel(&self) {
        let _ = std::fs::File::create(&self.cancel_file);
    }

    /// worker stderr 的最后几行；worker 没能自己报告错误时用来解释退出码。
    pub fn stderr_tail(&self) -> Option<String> {
        let text: String = self.stderr_tail.lock().unwrap().iter().cloned().collect::<Vec<_>>().join(" | ");
        let text: String = text.chars().take(400).collect();
        (!text.is_empty()).then_some(text)
    }

    /// 非阻塞收尸；返回 Some(退出码) 表示已结束（并已清理取消文件）。
    pub fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        let status = self.child.try_wait()?;
        if status.is_some() {
            self.reaped = true;
            // 进程已退出，stderr 的写端随之关闭，读取线程很快读到 EOF；上限防止意外挂住。
            let _ = self.stderr_done.recv_timeout(Duration::from_millis(200));
            let _ = std::fs::remove_file(&self.cancel_file);
        }
        Ok(status)
    }
}

impl Drop for Detection {
    fn drop(&mut self) {
        if !self.reaped {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        let _ = std::fs::remove_file(&self.cancel_file);
    }
}

pub fn model_path(repo_root: &Path) -> PathBuf {
    repo_root.join("models").join("transnet-split.onnx")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn exiting(text: &str, code: i32) -> Command {
        let mut command = Command::new("cmd");
        // cmd 的转义：括号里的整段交给 cmd，1>&2 把 echo 写到 stderr。
        command.args(["/C", &format!("({text}) 1>&2 & exit {code}")]);
        // 与正式 spawn 相同的管道布局。
        command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        command
    }

    #[test]
    fn prepare_cancel_removes_stale_marker() {
        let dir = std::env::temp_dir().join("cv-detect-test");
        std::fs::create_dir_all(&dir).unwrap();
        let index_json = dir.join("index.json");
        let stale = cancel_path(&index_json);
        std::fs::write(&stale, "stale").unwrap();
        let path = prepare_cancel(&index_json);
        assert_eq!(path, stale);
        assert!(!stale.exists(), "残留取消文件必须被清掉");
    }

    #[test]
    fn cancel_path_matches_worker_convention() {
        let path = cancel_path(Path::new("/tmp/x/index-123.json"));
        assert_eq!(path, Path::new("/tmp/x/cancel"));
    }

    #[test]
    fn worker_stderr_tail_survives_the_exit() {
        let dir = std::env::temp_dir().join("cv-detect-stderr-test");
        std::fs::create_dir_all(&dir).unwrap();
        let index_json = dir.join("index-stderr-test.json");
        let (tx, rx) = mpsc::channel();
        let mut job = Detection::spawn_command(exiting("echo boom-from-stderr", 1), &index_json, move |_| {
            let _ = tx.send(());
        })
        .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        let mut status = None;
        while status.is_none() && std::time::Instant::now() < deadline {
            status = job.try_wait().unwrap();
            if status.is_none() {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        assert_eq!(status.map(|s| s.code().unwrap_or(-1)), Some(1));
        assert!(rx.try_recv().is_err(), "这个进程不写 stdout");
        let tail = job.stderr_tail().expect("stderr 尾部必须保留");
        assert!(tail.contains("boom-from-stderr"), "实得：{tail}");
        assert!(!cancel_path(&index_json).exists());
    }
}
