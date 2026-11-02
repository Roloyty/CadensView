//! CadensView 二开 GUI：egui 前端 + FFmpeg 播放 + shotlab Python 边车。
//! 所有 RPC 调用都在工作线程执行，UI 线程只消费 UiMsg。

mod detect;
mod model;
mod player;
mod rpc;

use eframe::egui;
use egui::{ColorImage, TextureHandle};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use model::{IndexInfo, State};

enum UiMsg {
    State(State),
    IndexReady { index: IndexInfo, index_json_path: PathBuf },
    RpcError(String),
    DetectionEvent(Value),
    DetectionFinished { code: i32 },
    IndexJobStarted(u64),
    IndexJobFailed(String),
}

struct App {
    rpc: Arc<rpc::Rpc>,
    msg_tx: Sender<UiMsg>,
    msg_rx: std::sync::mpsc::Receiver<UiMsg>,
    /// 工作线程用它在消息送达后请求重绘；egui 空闲时不会自轮询。
    wake: player::Waker,
    state: State,
    index: Option<IndexInfo>,
    index_json_path: Option<PathBuf>,
    player: Option<player::PlayerHandle>,
    playback_marker: Option<player::PlaybackMarker>,
    texture: Option<TextureHandle>,
    pending_image: Option<(usize, usize, Vec<u8>)>,
    note_draft: String,
    note_shot: String,
    note_dirty: bool,
    ui_frame: u64,
    detection: Option<detect::Detection>,
    detection_out: Option<PathBuf>,
    detection_cpu_retried: bool,
    index_job: Option<u64>,
    index_busy: bool,
    repo_root: PathBuf,
    ffmpeg: PathBuf,
    python: PathBuf,
    status: String,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Result<Self, String> {
        let repo_root = find_repo_root();
        let python = rpc::python_path(&repo_root);
        if !python.exists() {
            return Err(format!("未找到 Python: {}（可用 CADENSVIEW_PY 指定）", python.display()));
        }
        let ffmpeg = repo_root.join("tools").join("ffmpeg.exe");
        if !ffmpeg.exists() {
            return Err(format!("未找到 ffmpeg: {}", ffmpeg.display()));
        }
        let rpc_handle = rpc::Rpc::spawn(&python, &repo_root)
            .map_err(|e| format!("边车启动失败: {e}（venv 依赖装了吗？）"))?;
        let (msg_tx, msg_rx) = std::sync::mpsc::channel();
        let ctx = cc.egui_ctx.clone();
        let wake: player::Waker = Arc::new(move || ctx.request_repaint());
        Ok(App {
            rpc: Arc::new(rpc_handle),
            msg_tx,
            msg_rx,
            wake,
            state: State::default(),
            index: None,
            index_json_path: None,
            player: None,
            playback_marker: None,
            texture: None,
            pending_image: None,
            note_draft: String::new(),
            note_shot: String::new(),
            note_dirty: false,
            ui_frame: 0,
            detection: None,
            detection_out: None,
            detection_cpu_retried: false,
            index_job: None,
            index_busy: false,
            repo_root,
            ffmpeg,
            python,
            status: "就绪。先打开视频或 .shotlab 项目。".into(),
        })
    }

    // -- 异步动作 ------------------------------------------------------------

    /// 结果不是 State 的调用（如 job.cancel）：只关心错误，成功静默。
    fn call_quiet(&self, method: &str, params: Value) {
        let method = method.to_string();
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        std::thread::spawn(move || {
            if let Err(error) = rpc.call(&method, params) {
                let _ = tx.send(UiMsg::RpcError(error));
            }
            wake();
        });
    }

    fn call(&self, method: &str, params: Value) {
        let method = method.to_string();
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        std::thread::spawn(move || {
            match rpc.call(&method, params) {
                Ok(Value::Null) => {}
                Ok(result) => {
                    if let Ok(state) = serde_json::from_value::<State>(result) {
                        let _ = tx.send(UiMsg::State(state));
                    }
                }
                Err(error) => {
                    let _ = tx.send(UiMsg::RpcError(error));
                }
            }
            // 结果送达后必须唤醒 UI，否则界面停在旧状态。
            wake();
        });
    }

    fn open_video(&mut self, video: PathBuf) {
        if self.index_busy {
            self.status = "索引任务进行中，先取消再打开新视频".into();
            return;
        }
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        self.status = "建立帧索引…（首次全量解码，耐心等待）".into();
        self.player = None;
        self.index_busy = true;
        std::thread::spawn(move || {
            // 走 job 通道：可取消、可显示进度，sidecar 也不会被阻塞。
            let reply = rpc.call("job.start", json!({
                "method": "index.build",
                "params": {"video_path": video.to_string_lossy()},
            }));
            match reply {
                Ok(value) => {
                    if let Some(job_id) = value.get("job_id").and_then(Value::as_u64) {
                        let _ = tx.send(UiMsg::IndexJobStarted(job_id));
                    }
                }
                Err(error) => {
                    let _ = tx.send(UiMsg::IndexJobFailed(error));
                }
            }
            wake();
        });
    }

    fn open_project(&mut self, path: PathBuf) {
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        self.status = "打开项目…".into();
        self.player = None;
        std::thread::spawn(move || {
            if let Err(error) = rpc.call("project.open", json!({"path": path.to_string_lossy()})) {
                let _ = tx.send(UiMsg::RpcError(error));
                wake();
                return;
            }
            match rpc.call("index.get", json!({})) {
                Ok(value) => {
                    // 项目已打开，不要再 project.new；索引走只读收尾。
                    if let Err(error) =
                        finish_index(Arc::clone(&rpc), tx.clone(), value.get("index").cloned().unwrap_or(Value::Null), Some(()))
                    {
                        let _ = tx.send(UiMsg::RpcError(error));
                    }
                }
                Err(error) => {
                    let _ = tx.send(UiMsg::RpcError(error));
                }
            }
            wake();
        });
    }

    fn save_project(&mut self, save_as: bool) {
        if save_as || self.state.project_path.is_none() {
            let file = rfd::FileDialog::new()
                .add_filter("CadensView 项目", &["shotlab"])
                .set_file_name("project.shotlab")
                .save_file();
            match file {
                Some(path) => self.call("project.save_as", json!({"path": path.to_string_lossy()})),
                None => return,
            }
        } else {
            self.call("project.save", json!({}));
        }
    }

    fn run_detection(&mut self, force_cpu: bool) {
        let (index_json, out_dir) = match (&self.index_json_path, unique_dir("det")) {
            (Some(path), Some(out)) => (path.clone(), out),
            _ => return,
        };
        let python = self.python.clone();
        let repo_root = self.repo_root.clone();
        let weights = detect::model_path(&self.repo_root);
        let tx_events = self.msg_tx.clone();
        let wake_events = Arc::clone(&self.wake);
        self.status = "切镜检测启动…".into();
        self.detection_out = Some(out_dir.clone());
        match detect::Detection::spawn(
            &python,
            &repo_root,
            &index_json,
            &out_dir,
            &weights,
            force_cpu,
            move |event| {
                let _ = tx_events.send(UiMsg::DetectionEvent(event));
                wake_events();
            },
        ) {
            Ok(job) => self.detection = Some(job),
            Err(error) => self.status = format!("检测启动失败: {error}"),
        }
    }

    fn select_shot(&mut self, shot_id: &str) {
        self.call("edit.select", json!({"shot_id": shot_id}));
        if let (Some(index), Some(row)) = (&self.index, self.state.shots.iter().find(|s| s.id == shot_id)) {
            if let Some(player) = &self.player {
                player.show(row.start.min(index.frames().saturating_sub(1)));
            }
        }
    }

    fn commit_cursor(&self, frame: u64) {
        self.call("edit.select", json!({"frame": frame}));
    }

    fn commit_note(&mut self) {
        if self.note_dirty && !self.note_shot.is_empty() {
            let shot = self.note_shot.clone();
            let text = self.note_draft.clone();
            self.call("edit.set_note", json!({"shot_id": shot, "text": text}));
            self.note_dirty = false;
        }
    }

    // -- 消息泵 ---------------------------------------------------------------

    fn pump(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.msg_rx.try_recv() {
            match msg {
                UiMsg::State(state) => self.state = state,
                UiMsg::IndexReady { index, index_json_path } => {
                    if let Some(video) = self.state.video_path.clone() {
                        let handle = player::PlayerHandle::spawn(
                            self.ffmpeg.clone(),
                            PathBuf::from(video),
                            index.clone(),
                            Arc::clone(&self.wake),
                        );
                        handle.show(0);
                        self.player = Some(handle);
                        self.playback_marker =
                            Some(player::PlaybackMarker::new(index_json_path.parent().unwrap()));
                        self.texture = None;
                    }
                    self.index = Some(index);
                    self.index_json_path = Some(index_json_path);
                }
                UiMsg::IndexJobStarted(job_id) => self.index_job = Some(job_id),
                UiMsg::IndexJobFailed(error) => {
                    self.index_busy = false;
                    self.status = format!("⚠ {error}");
                }
                UiMsg::RpcError(error) => self.status = format!("⚠ {error}"),
                UiMsg::DetectionEvent(event) => self.on_detection_event(event),
                UiMsg::DetectionFinished { code } => {
                    self.detection = None;
                    if code == 0 {
                        self.status = "检测完成，写入镜头…".into();
                        if let Some(out_dir) = self.detection_out.clone() {
                            self.call(
                                "project.apply_detection",
                                json!({"result_dir": out_dir.to_string_lossy()}),
                            );
                        }
                    } else if code == 2 {
                        self.status = "检测已取消".into();
                    } else if code == 3 && !self.detection_cpu_retried {
                        // 协议约定：GPU 失败后在纯 CPU 进程重试一次。
                        self.detection_cpu_retried = true;
                        self.status = "GPU 推理失败，切换 CPU 重试…".into();
                        self.run_detection(true);
                    } else {
                        self.status = format!("检测失败（退出码 {code}）");
                    }
                }
            }
        }
        // 播放器帧消息走独立通道（不在 UiMsg 里）：先收全，再逐个处理避免借用冲突。
        let mut player_messages = Vec::new();
        if let Some(player) = &self.player {
            while let Ok(message) = player.msgs.try_recv() {
                player_messages.push(message);
            }
        }
        for message in player_messages {
            self.on_player_msg(message);
        }
        // 边车进度事件（同步 handler 的 emit 与 job 事件都在这里落地）。
        for event in self.rpc.drain_events() {
            let topic = event
                .get("topic")
                .or_else(|| event.get("job"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let phase = event.get("phase").and_then(Value::as_str).unwrap_or("progress");
            let job_id = event.get("job_id").and_then(Value::as_u64);
            match (topic, phase) {
                ("index", "progress") => {
                    if let Some(done) = event.get("done").and_then(Value::as_u64) {
                        let time_s = event.get("time_s").and_then(Value::as_f64).unwrap_or(0.0);
                        self.status = format!("建立帧索引… 已解码 {done} 帧（视频时间 {time_s:.0}s）");
                    }
                }
                ("index.build", "done") if job_id == self.index_job || self.index_busy => {
                    self.index_busy = false;
                    self.index_job = None;
                    let index_value = event
                        .pointer("/result/index")
                        .cloned()
                        .unwrap_or(Value::Null);
                    let rpc = Arc::clone(&self.rpc);
                    let tx = self.msg_tx.clone();
                    let wake = Arc::clone(&self.wake);
                    self.status = "索引完成，装配项目…".into();
                    std::thread::spawn(move || {
                        // finish_index 内部会再发 RPC（project.new / state），不能在 UI 线程跑。
                        if let Err(error) = finish_index(rpc, tx.clone(), index_value, None) {
                            let _ = tx.send(UiMsg::IndexJobFailed(error));
                        }
                        wake();
                    });
                }
                ("index.build", "error") => {
                    self.index_busy = false;
                    self.index_job = None;
                    let message = event.get("message").and_then(Value::as_str).unwrap_or("未知");
                    let kind = event.get("kind").and_then(Value::as_str).unwrap_or("");
                    self.status = if kind.contains("Cancelled") {
                        "索引已取消".into()
                    } else {
                        format!("索引失败: {message}")
                    };
                }
                (_, "error") => {
                    let message = event.get("message").and_then(Value::as_str).unwrap_or("未知");
                    self.status = format!("后台任务失败: {message}");
                }
                _ => {}
            }
        }
        // 纹理上传必须在 UI 线程（需要 egui Context）
        if let Some((width, height, rgba)) = self.pending_image.take() {
            let image = ColorImage::from_rgba_unmultiplied([width, height], &rgba);
            match self.texture.as_mut() {
                Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
                None => {
                    self.texture = Some(ctx.load_texture("video", image, egui::TextureOptions::LINEAR))
                }
            }
        }
        // 检测收尸轮询
        if let Some(job) = self.detection.as_mut() {
            if let Ok(Some(status)) = job.try_wait() {
                let code = status.code().unwrap_or(-1);
                let _ = self.msg_tx.send(UiMsg::DetectionFinished { code });
            } else {
                ctx.request_repaint_after(Duration::from_millis(200));
            }
        }
        if self.player.as_ref().map(player::PlayerHandle::is_playing).unwrap_or(false) {
            ctx.request_repaint();
        }
    }

    fn on_player_msg(&mut self, message: player::PlayerMsg) {
        match message {
            player::PlayerMsg::Frame { frame, width, height, rgba } => {
                self.pending_image = Some((width, height, rgba));
                self.ui_frame = frame;
                if let Some(marker) = self.playback_marker.as_mut() {
                    if let Some(player) = &self.player {
                        marker.set(player.is_playing());
                    }
                }
            }
            player::PlayerMsg::Ended => {
                if let Some(marker) = self.playback_marker.as_mut() {
                    marker.set(false);
                }
                let frame = self.player.as_ref().map(|p| p.frame()).unwrap_or(0);
                self.commit_cursor(frame);
            }
            player::PlayerMsg::Failed(error) => self.status = format!("播放: {error}"),
        }
    }

    fn on_detection_event(&mut self, event: Value) {
        // worker 事件字段是 type（analysis_worker.emit），保留 kind/event 兜底。
        let kind = event
            .get("type")
            .or_else(|| event.get("kind"))
            .or_else(|| event.get("event"))
            .and_then(Value::as_str)
            .unwrap_or("");
        match kind {
            "loading" => self.status = "检测进程启动，加载模型…".into(),
            "progress" => {
                let frame = event.get("frame").and_then(Value::as_u64).unwrap_or(0);
                let total = event.get("total").and_then(Value::as_u64).unwrap_or(0);
                self.status = format!("检测中 {frame}/{total} 帧");
            }
            "partial" => {
                let count = event.get("boundaries").and_then(Value::as_array).map(Vec::len).unwrap_or(0);
                self.status = format!("检测中…已发现 {count} 个切点");
            }
            "device" => {
                let device = event.get("device").and_then(Value::as_str).unwrap_or("?");
                self.status = format!("检测设备: {device}");
            }
            "complete" => self.status = "检测收尾…".into(),
            "cancelled" => self.status = "检测已取消".into(),
            "error" => {
                let message = event.get("message").and_then(Value::as_str).unwrap_or("未知");
                self.status = format!("检测错误: {message}");
            }
            _ => {}
        }
    }
}

// -- 索引就绪的公共收尾 --------------------------------------------------------

/// 在后台线程完成索引收尾；所有发出的消息由调用方负责唤醒 UI（palyer::Waker）。
fn finish_index(
    rpc: Arc<rpc::Rpc>,
    tx: Sender<UiMsg>,
    index_value: Value,
    existing_project: Option<()>,
) -> Result<(), String> {
    let index: IndexInfo = serde_json::from_value(index_value.clone())
        .map_err(|e| format!("索引解析失败: {e}"))?;
    if !index.complete {
        return Err("索引不完整（limit 前缀）".into());
    }
    let dir = std::env::temp_dir().join("cadensview");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let index_json_path = dir.join(format!(
        "index-{}.json",
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis()
    ));
    let payload = serde_json::to_string(&index_value).map_err(|e| e.to_string())?;
    std::fs::write(&index_json_path, payload).map_err(|e| e.to_string())?;
    if existing_project.is_none() {
        let state = rpc.call("project.new", json!({}))?;
        let _ = tx.send(UiMsg::State(serde_json::from_value(state).unwrap_or_default()));
    } else {
        let state = rpc.call("project.state", json!({}))?;
        let _ = tx.send(UiMsg::State(serde_json::from_value(state).unwrap_or_default()));
    }
    tx.send(UiMsg::IndexReady { index, index_json_path })
        .map_err(|_| "UI 已退出".to_string())?;
    Ok(())
}

/// 生成唯一的输出目录**路径**（不创建目录）。
/// worker 的结果目录必须由它独占创建（核心 `save_result(exist_ok=False)` 不变式），
/// GUI 预创建会让检测在保存阶段必然 WinError 183。
fn unique_dir(prefix: &str) -> Option<PathBuf> {
    use std::sync::atomic::AtomicU64;
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let base = std::env::temp_dir().join("cadensview");
    std::fs::create_dir_all(&base).ok()?;
    for _ in 0..16 {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
        let ticket = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = base.join(format!("{prefix}-{stamp}-{ticket}"));
        if !dir.exists() {
            return Some(dir);
        }
    }
    None
}

fn find_repo_root() -> PathBuf {
    if let Ok(root) = std::env::var("CADENSVIEW_ROOT") {
        return PathBuf::from(root);
    }
    if let Ok(exe) = std::env::current_exe() {
        for dir in exe.ancestors().skip(1) {
            if dir.join("shotlab").is_dir() && dir.join(".venv").is_dir() {
                return dir.to_path_buf();
            }
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap()
}

// -- UI ------------------------------------------------------------------

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.pump(&ctx);
        egui::Panel::top("topbar").show(ui, |ui| self.topbar(ui));
        egui::Panel::bottom("statusbar").show(ui, |ui| {
            ui.horizontal(|ui| {
                if self.state.dirty {
                    ui.colored_label(egui::Color32::ORANGE, "● 未保存");
                }
                ui.label(&self.status);
            });
        });
        egui::Panel::left("shots").default_size(230.0).show(ui, |ui| self.shot_list(ui));
        egui::Panel::right("notes").default_size(290.0).show(ui, |ui| self.note_panel(ui));
        egui::CentralPanel::default().show(ui, |ui| self.player_panel(ui));
    }
}

impl App {
    fn topbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let idle = !self.index_busy;
            if ui.add_enabled(idle, egui::Button::new("打开视频")).clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("视频", &["mkv", "mp4", "avi", "mov", "webm", "ts", "m2ts", "flv"])
                    .pick_file()
                {
                    self.open_video(path);
                }
            }
            if ui.add_enabled(!self.index_busy, egui::Button::new("打开项目")).clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("CadensView 项目", &["shotlab"])
                    .pick_file()
                {
                    self.open_project(path);
                }
            }
            ui.separator();
            if ui.add_enabled(!self.index_busy, egui::Button::new("保存")).clicked() {
                self.save_project(false);
            }
            if ui.add_enabled(!self.index_busy, egui::Button::new("另存为")).clicked() {
                self.save_project(true);
            }
            ui.separator();
            if self.index_busy {
                if ui.button("取消索引").clicked() {
                    if let Some(job_id) = self.index_job {
                        self.call_quiet("job.cancel", json!({"job_id": job_id}));
                    }
                }
                ui.spinner();
                ui.label("索引进行中…");
            } else {
                let ready = self.index.is_some() && self.detection.is_none() && self.state.pending_start.is_some();
                if ui.add_enabled(ready, egui::Button::new("检测切镜")).clicked() {
                    self.detection_cpu_retried = false;
                    self.run_detection(false);
                }
            }
            if self.detection.is_some() && ui.button("取消检测").clicked() {
                if let Some(job) = &self.detection {
                    job.cancel();
                }
            }
            ui.separator();
            if ui.add_enabled(self.state.can_undo, egui::Button::new("↶ 撤销")).clicked() {
                self.call("edit.undo", json!({}));
            }
            if ui.add_enabled(self.state.can_redo, egui::Button::new("↷ 重做")).clicked() {
                self.call("edit.redo", json!({}));
            }
            if self.state.pending_start.is_some() && self.index.is_some() {
                ui.colored_label(egui::Color32::GRAY, "（有待分析尾段：先跑检测）");
            }
        });
    }

    fn shot_list(&mut self, ui: &mut egui::Ui) {
        ui.heading("镜头");
        let mut follow = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            for shot in &self.state.shots {
                let selected = shot.id == self.state.selected_id;
                let label = format!(
                    "#{} {}帧 · {:.2}s{}",
                    shot.number,
                    shot.end - shot.start,
                    shot.duration_s,
                    if shot.pending { " · 待分析" } else { "" }
                );
                let text = if shot.pending {
                    egui::RichText::new(label).color(egui::Color32::GRAY)
                } else {
                    egui::RichText::new(label)
                };
                if ui.selectable_label(selected, text).clicked() {
                    follow = Some(shot.id.clone());
                }
                if !shot.note.is_empty() {
                    ui.indent(format!("note-{}", shot.id), |ui| {
                        ui.set_max_width(ui.available_width());
                        ui.small(egui::RichText::new(truncate(&shot.note, 60)).weak());
                    });
                }
            }
        });
        if let Some(id) = follow {
            self.select_shot(&id);
        }
    }

    fn note_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("镜头笔记");
        let selected = self
            .state
            .shots
            .iter()
            .find(|s| s.id == self.state.selected_id)
            .cloned();
        match selected {
            Some(shot) => {
                if self.note_shot != shot.id {
                    self.commit_note();
                    self.note_shot = shot.id.clone();
                    self.note_draft = shot.note.clone();
                    self.note_dirty = false;
                }
                ui.label(format!("镜头 #{}（{}–{} 帧）", shot.number, shot.start, shot.end));
                let response = egui::TextEdit::multiline(&mut self.note_draft)
                    .desired_rows(8)
                    .desired_width(f32::INFINITY)
                    .show(ui);
                if response.response.changed() {
                    self.note_dirty = true;
                }
                if response.response.lost_focus() && self.note_dirty {
                    self.commit_note();
                }
                ui.separator();
                let current = self.player.as_ref().map(|p| p.frame()).unwrap_or(self.ui_frame);
                let in_shot = current >= shot.start && current < shot.end;
                if ui
                    .add_enabled(in_shot && !shot.pending, egui::Button::new("在此帧拆分"))
                    .clicked()
                {
                    self.call("edit.split", json!({"frame": current}));
                }
                if ui
                    .add_enabled(shot.number + 1 < self.state.completed_count as usize, egui::Button::new("与左镜头合并"))
                    .clicked()
                {
                    self.call("edit.merge", json!({"row": shot.number}));
                }
            }
            None => {
                ui.label("（未选中镜头）");
            }
        }
    }

    fn player_panel(&mut self, ui: &mut egui::Ui) {
        let frames = self.index.as_ref().map(|i| i.frames()).unwrap_or(1).max(1);
        match &self.texture {
            Some(texture) => {
                let size = texture.size_vec2();
                ui.add(
                    egui::Image::new(egui::load::SizedTexture::new(texture.id(), size))
                        .max_width(ui.available_width())
                        .maintain_aspect_ratio(true),
                );
            }
            None => {
                ui.centered_and_justified(|ui| ui.label("打开视频后在此播放（静音分析模式）"));
            }
        }
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let playing = self.player.as_ref().map(|p| p.is_playing()).unwrap_or(false);
            let play_label = if playing { "⏸" } else { "▶" };
            if ui.add_enabled(self.player.is_some(), egui::Button::new(play_label)).clicked() {
                if let Some(player) = &self.player {
                    if playing {
                        let frame = player.frame();
                        player.stop();
                        player.show(frame);
                        self.commit_cursor(frame);
                    } else {
                        player.play(self.ui_frame.min(frames - 1));
                    }
                }
            }
            if ui
                .add_enabled(self.player.is_some(), egui::Button::new("◀ 1帧"))
                .clicked()
            {
                self.step(-1, frames);
            }
            if ui
                .add_enabled(self.player.is_some(), egui::Button::new("1帧 ▶"))
                .clicked()
            {
                self.step(1, frames);
            }
            ui.separator();
            let mut slider = self.ui_frame.min(frames - 1) as i64;
            let response = ui.add(
                egui::Slider::new(&mut slider, 0..=(frames as i64 - 1))
                    .show_value(true)
                    .text("帧"),
            );
            if response.changed() {
                self.ui_frame = slider.max(0) as u64;
                if let Some(player) = &self.player {
                    player.show(self.ui_frame);
                }
            }
            if response.drag_stopped() {
                self.commit_cursor(self.ui_frame);
            }
            let seconds = self.index.as_ref().map(|i| i.seconds(self.ui_frame)).unwrap_or(0.0);
            ui.monospace(format!("{:.2}s", seconds));
        });
    }

    fn step(&mut self, delta: i64, frames: u64) {
        if let Some(player) = &self.player {
            let target = (player.frame() as i64 + delta).clamp(0, frames as i64 - 1) as u64;
            player.show(target);
            self.ui_frame = target;
            self.commit_cursor(target);
        }
    }
}

fn truncate(text: &str, max: usize) -> String {
    let cleaned: String = text.chars().filter(|c| !c.is_control()).collect();
    if cleaned.chars().count() > max {
        let mut shortened: String = cleaned.chars().take(max).collect();
        shortened.push('…');
        shortened
    } else {
        cleaned
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("CadensView（二开版）"),
        ..Default::default()
    };
    eframe::run_native(
        "cadensview",
        options,
        Box::new(|cc| match App::new(cc) {
            Ok(app) => Ok(Box::new(app) as Box<dyn eframe::App>),
            Err(error) => Err(error.into()),
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_respects_char_boundaries() {
        assert_eq!(truncate("短", 5), "短");
        assert_eq!(truncate("一二三四五六", 3), "一二三…");
        assert_eq!(truncate("a\nb\tc", 10), "abc");
    }

    #[test]
    fn unique_dir_is_unique() {
        let first = unique_dir("test").expect("dir");
        let second = unique_dir("test").expect("dir");
        assert_ne!(first, second);
        let _ = std::fs::remove_dir_all(&first);
        let _ = std::fs::remove_dir_all(&second);
    }
}
