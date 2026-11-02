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

struct Shared {
    stdin: Mutex<std::process::ChildStdin>,
    pending: Mutex<HashMap<u64, std::sync::mpsc::Sender<RpcReply>>>,
    events: Mutex<Vec<Value>>,
}

pub struct Rpc {
    shared: Arc<Shared>,
    next_id: AtomicU64,
    _child: Child, // kept alive; drop kills the sidecar
}

impl Rpc {
    pub fn spawn(python: &Path, repo_root: &Path) -> std::io::Result<Rpc> {
        let mut child = Command::new(python)
            .current_dir(repo_root)
            .args(["-m", "shotlab.rpc"])
            .env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = child.stdin.take().expect("sidecar stdin");
        let stdout = child.stdout.take().expect("sidecar stdout");
        let shared = Arc::new(Shared {
            stdin: Mutex::new(stdin),
            pending: Mutex::new(HashMap::new()),
            events: Mutex::new(Vec::new()),
        });
        let reader_shared = Arc::clone(&shared);
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                let Ok(value) = serde_json::from_str::<Value>(&line) else { continue };
                if let Some(id) = value.get("id").and_then(Value::as_u64) {
                    let sender = reader_shared.pending.lock().unwrap().remove(&id);
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
        });
        Ok(Rpc { shared, next_id: AtomicU64::new(1), _child: child })
    }

    /// Blocking call; run from a worker thread, never from the UI thread.
    pub fn call(&self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = std::sync::mpsc::channel();
        self.shared.pending.lock().unwrap().insert(id, tx);
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
            Err(_) => Err("sidecar 响应超时或已退出".into()),
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
