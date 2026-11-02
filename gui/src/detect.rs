//! 切镜检测：spawn `python -m shotlab.onnx_worker`（既有子进程协议）。
//! stdout JSON-lines: loading/device/progress/partial/complete/cancelled/error；
//! 取消 = 在 index JSON 同目录创建 cancel 文件（worker 侧协议），
//! 退出码 3 = GPU 错误，GUI 可在纯 CPU 新进程重试一次。
//!
//! cancel 文件的生命周期必须严格管理：spawn 前清残留、结束后删除、
//! Drop 时兜底删除，否则一次取消会永久污染后续所有检测。

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

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

pub struct Detection {
    pub cancel_file: PathBuf,
    child: Child,
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
        let cancel_file = prepare_cancel(index_json);
        let mut child = Command::new(python)
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
            .stderr(Stdio::null())
            .spawn()?;
        let stdout = child.stdout.take().expect("worker stdout");
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                let Ok(value) = serde_json::from_str::<Value>(&line) else { continue };
                on_event(value);
            }
        });
        Ok(Detection { cancel_file, child, reaped: false })
    }

    /// 请求取消：直接落地哨兵文件，worker 每 50ms 检查一次。
    pub fn cancel(&self) {
        let _ = std::fs::File::create(&self.cancel_file);
    }

    /// 非阻塞收尸；返回 Some(退出码) 表示已结束（并已清理取消文件）。
    pub fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        let status = self.child.try_wait()?;
        if status.is_some() {
            self.reaped = true;
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
}