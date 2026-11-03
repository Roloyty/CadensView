//! Python 边车客户端：stdin/stdout 按行 JSON-RPC。
//! 请求顺序执行；进度事件走独立通道，长任务不阻塞 UI 线程。

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde_json::{json, Value};

pub struct RpcReply {
    pub ok: bool,
    pub result: Value,
    pub error: String,
}

type Pending = HashMap<u64, std::sync::mpsc::Sender<RpcReply>>;

struct Shared {
    stdin: Mutex<std::process::ChildStdin>,
    /// `None` once the sidecar's stdout closed: waiting callers fail at once instead of timing out.
    pending: Mutex<Option<Pending>>,
    events: Mutex<Vec<Value>>,
}

pub struct Rpc {
    shared: Arc<Shared>,
    next_id: AtomicU64,
    // Kept for ownership only; dropping a Child does not kill it. The sidecar exits when
    // its stdin (held by `shared`) closes.
    _child: Child,
}

impl Rpc {
    pub fn spawn(python: &Path, repo_root: &Path) -> std::io::Result<Rpc> {
        let mut command = Command::new(python);
        command
            .current_dir(repo_root)
            .args(["-m", "shotlab.rpc"])
            .env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8");
        Self::spawn_command(command, Some(&log_path()))
    }

    fn spawn_command(mut command: Command, log: Option<&Path>) -> std::io::Result<Rpc> {
        // Python 异常栈写日志文件，边车崩溃后可追查；打不开日志时退回丢弃。
        let stderr = log.and_then(|path| std::fs::File::create(path).ok())
            .map(Stdio::from).unwrap_or_else(Stdio::null);
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr)
            .spawn()?;
        let stdin = child.stdin.take().expect("sidecar stdin");
        let stdout = child.stdout.take().expect("sidecar stdout");
        let shared = Arc::new(Shared {
            stdin: Mutex::new(stdin),
            pending: Mutex::new(Some(HashMap::new())),
            events: Mutex::new(Vec::new()),
        });
        let reader_shared = Arc::clone(&shared);
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                let Ok(value) = serde_json::from_str::<Value>(&line) else { continue };
                if let Some(id) = value.get("id").and_then(Value::as_u64) {
                    let sender = reader_shared.pending.lock().unwrap().as_mut().and_then(|p| p.remove(&id));
                    if let Some(sender) = sender {
                        let ok = value.get("ok").and_then(Value::as_bool).unwrap_or(false);
                        let _ = sender.send(RpcReply {
                            ok,
                            result: value.get("result").cloned().unwrap_or(Value::Null),
                            error: value
                                .pointer("/error/message")
                                .and_then(Value::as_str)
                                .unwrap_or("unknown error")
                                .to_string(),
                        });
                    }
                } else if value.get("event").is_some() {
                    reader_shared.events.lock().unwrap().push(value);
                }
            }
            // 边车退出：丢弃全部 Sender，等待中的 recv 立即断开。
            reader_shared.pending.lock().unwrap().take();
        });
        Ok(Rpc { shared, next_id: AtomicU64::new(1), _child: child })
    }

    /// Blocking call; run from a worker thread, never from the UI thread.
    pub fn call(&self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = std::sync::mpsc::channel();
        match self.shared.pending.lock().unwrap().as_mut() {
            Some(pending) => { pending.insert(id, tx); }
            None => return Err(format!("sidecar 已退出，详见 {}", log_path().display())),
        }
        let request = json!({"id": id, "method": method, "params": params});
        let line = serde_json::to_string(&request).map_err(|e| format!("请求序列化失败: {e}"))?;
        {
            let mut stdin = self.shared.stdin.lock().unwrap();
            stdin
                .write_all((line + "\n").as_bytes())
                .and_then(|_| stdin.flush())
                .map_err(|e| format!("sidecar 写入失败: {e}"))?;
        }
        match rx.recv_timeout(Duration::from_secs(3600)) {
            Ok(reply) if reply.ok => Ok(reply.result),
            Ok(reply) => Err(reply.error),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) =>
                Err(format!("sidecar 已退出，详见 {}", log_path().display())),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if let Some(pending) = self.shared.pending.lock().unwrap().as_mut() { pending.remove(&id); }
                Err("sidecar 响应超时".into())
            }
        }
    }

    pub fn drain_events(&self) -> Vec<Value> {
        let mut queue = self.shared.events.lock().unwrap();
        std::mem::take(&mut *queue)
    }
}

/// Locate the venv interpreter: env override, then relative to the repo root.
pub fn python_path(repo_root: &Path) -> PathBuf {
    if let Ok(path) = std::env::var("CADENSVIEW_PY") {
        return PathBuf::from(path);
    }
    repo_root.join(".venv").join("Scripts").join("python.exe")
}

/// Sidecar stderr log, overwritten at each start.
pub fn log_path() -> PathBuf {
    std::env::temp_dir().join("cadensview-sidecar.log")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn exiting_process() -> Command {
        // 读完 stdin 前就退出，模拟边车崩溃。
        if cfg!(windows) {
            let mut command = Command::new("cmd");
            command.args(["/C", "exit", "3"]);
            command
        } else {
            let mut command = Command::new("sh");
            command.args(["-c", "exit 3"]);
            command
        }
    }

    #[test]
    fn calls_fail_fast_after_the_sidecar_exits() {
        let rpc = Rpc::spawn_command(exiting_process(), None).unwrap();
        let started = Instant::now();
        let first = rpc.call("ping", json!({}));
        let second = rpc.call("ping", json!({}));
        assert!(first.is_err() && second.is_err());
        assert!(started.elapsed() < Duration::from_secs(10), "took {:?}", started.elapsed());
    }
}
