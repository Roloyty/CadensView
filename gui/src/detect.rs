//! 切镜检测：spawn `python -m shotlab.onnx_worker`（既有子进程协议）。
//! stdout JSON-lines: loading/device/progress/partial/complete/cancelled/error；
//! 取消 = 在 index 目录创建 cancel 文件；退出码 3 = GPU 错误可 CPU 重试。

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::Value;

pub struct Detection {
    pub cancel: Arc<AtomicBool>,
    child: Child,
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
        std::fs::create_dir_all(out_dir)?;
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
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_flag = Arc::clone(&cancel);
        let cancel_dir = index_json.parent().map(Path::to_path_buf);
        std::thread::spawn(move || {
            // 取消哨兵：cancel 文件是 worker 侧的协议开关。
            while !cancel_flag.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            if let Some(dir) = cancel_dir.as_ref() {
                let _ = std::fs::File::create(dir.join("cancel"));
            }
        });
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                let Ok(value) = serde_json::from_str::<Value>(&line) else { continue };
                on_event(value);
            }
        });
        Ok(Detection { cancel, child })
    }

    /// 非阻塞收尸：返回 Some(退出码) 表示已结束。
    pub fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.child.try_wait()
    }
}

pub fn model_path(repo_root: &Path) -> PathBuf {
    repo_root.join("models").join("transnet-split.onnx")
}
