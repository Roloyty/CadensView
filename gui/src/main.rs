//! CadensView 二开 GUI：egui 前端 + FFmpeg 播放 + shotlab Python 边车。
//! 所有 RPC 调用都在工作线程执行，UI 线程只消费 UiMsg。

mod detect;
mod model;
mod player;
mod rpc;
mod thumb;

use eframe::egui;
use egui::{ColorImage, TextureHandle};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use model::{BarcodeSample, IndexInfo, PaletteInfo, State, StatsData};

enum UiMsg {
    State(State),
    IndexReady { index: IndexInfo, index_json_path: PathBuf },
    RpcError(String),
    DetectionEvent(Value),
    DetectionFinished { code: i32 },
    IndexJobStarted(u64),
    IndexJobFailed(String),
    StatsReady(StatsData),
    PaletteReady { shot_id: String, palette: PaletteInfo },
    AnalysisJobStarted(u64),
    AnalysisDone(Value),
    AnalysisFailed(String),
    Thumb(thumb::ThumbOutcome),
    AbCaptured { slot: usize, frame: u64, width: usize, height: usize, rgba: Vec<u8> },
    ExportSaved(String),
}

/// 右侧面板的三个页签。
#[derive(Clone, Copy, PartialEq)]
enum RightTab {
    Notes,
    Stats,
    Colors,
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
    // -- 统计面板 ------------------------------------------------------------
    right_tab: RightTab,
    stats: Option<StatsData>,
    stats_stale: bool,
    stats_loading: bool,
    // -- 色彩面板 ------------------------------------------------------------
    palette: Option<PaletteInfo>,
    palette_shot: String,
    palette_loading: bool,
    barcode: Vec<BarcodeSample>,
    analysis_job: Option<u64>,
    // -- 缩略图 --------------------------------------------------------------
    thumb_tx: Option<Sender<thumb::ThumbRequest>>,
    thumbs: HashMap<String, TextureHandle>,
    thumb_pending: HashSet<String>,
    thumb_failed: HashSet<String>,
    // -- A/B 对照 ------------------------------------------------------------
    ab_textures: [Option<TextureHandle>; 2],
    ab_data: [Option<Vec<u8>>; 2],
    ab_sizes: [(usize, usize); 2],
    ab_frames: [Option<u64>; 2],
    ab_show: bool,
    ab_overlay: bool,
    ab_alpha: f32,
    ab_blend: Option<TextureHandle>,
    ab_blend_dirty: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Result<Self, String> {
        install_cjk_fonts(&cc.egui_ctx);
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
        let mut app = App {
            rpc: Arc::new(rpc_handle),
            msg_tx: msg_tx.clone(),
            msg_rx,
            wake: Arc::clone(&wake),
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
            right_tab: RightTab::Notes,
            stats: None,
            stats_stale: false,
            stats_loading: false,
            palette: None,
            palette_shot: String::new(),
            palette_loading: false,
            barcode: Vec::new(),
            analysis_job: None,
            thumb_tx: None,
            thumbs: HashMap::new(),
            thumb_pending: HashSet::new(),
            thumb_failed: HashSet::new(),
            ab_textures: [None, None],
            ab_data: [None, None],
            ab_sizes: [(0, 0), (0, 0)],
            ab_frames: [None, None],
            ab_show: false,
            ab_overlay: false,
            ab_alpha: 0.5,
            ab_blend: None,
            ab_blend_dirty: false,
        };
        // 缩略图 worker：消费 ThumbOutcome 后发到主消息泵。
        let (thumb_done_tx, thumb_done_rx) = std::sync::mpsc::channel::<thumb::ThumbOutcome>();
        let thumb_tx = thumb::spawn_worker(app.ffmpeg.clone(), thumb_done_tx, Arc::clone(&wake));
        app.thumb_tx = Some(thumb_tx);
        // 缩略图完成转发线程（channel → UiMsg 泵）。
        let forward_tx = msg_tx.clone();
        std::thread::spawn(move || {
            while let Ok(outcome) = thumb_done_rx.recv() {
                let _ = forward_tx.send(UiMsg::Thumb(outcome));
            }
        });
        Ok(app)
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
                UiMsg::State(state) => {
                    // 镜头结构变化（数量/边界）→ 统计与选中镜头调色板失效。
                    let structure_changed = state.shots.len() != self.state.shots.len()
                        || state.shots.iter().zip(self.state.shots.iter())
                            .any(|(new, old)| new.start != old.start || new.end != old.end);
                    if structure_changed {
                        self.stats_stale = true;
                        self.palette = None;
                        self.palette_shot.clear();
                    }
                    self.state = state;
                }
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
                UiMsg::StatsReady(stats) => {
                    self.stats = Some(stats);
                    self.stats_loading = false;
                    self.stats_stale = false;
                }
                UiMsg::PaletteReady { shot_id, palette } => {
                    // 迟到的调色板（用户已切换镜头）直接丢弃。
                    if shot_id == self.palette_shot || shot_id.is_empty() {
                        self.palette = Some(palette);
                    }
                    self.palette_loading = false;
                }
                UiMsg::AnalysisJobStarted(job_id) => self.analysis_job = Some(job_id),
                UiMsg::AnalysisDone(value) => {
                    self.analysis_job = None;
                    let samples: Vec<BarcodeSample> = value
                        .pointer("/video_color_timeline/movie_barcode")
                        .and_then(|v| serde_json::from_value(v.clone()).ok())
                        .unwrap_or_default();
                    self.barcode = samples;
                    // 分析结果附带最新统计，顺手刷新统计面板。
                    if let Some(stats) = value.get("shot_statistics") {
                        if let Ok(parsed) = serde_json::from_value::<StatsData>(stats.clone()) {
                            self.stats = Some(parsed);
                            self.stats_stale = false;
                        }
                    }
                    self.status = "色彩分析完成".into();
                }
                UiMsg::AnalysisFailed(error) => {
                    self.analysis_job = None;
                    self.status = format!("色彩分析: {error}");
                }
                UiMsg::Thumb(thumb::ThumbOutcome::Ready { key, width, height, rgba }) => {
                    self.thumb_pending.remove(&key);
                    let image = ColorImage::from_rgba_unmultiplied(
                        [width as usize, height as usize],
                        &rgba,
                    );
                    match self.thumbs.get_mut(&key) {
                        Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
                        None => {
                            self.thumbs.insert(
                                key.clone(),
                                ctx.load_texture(format!("thumb-{key}"), image, egui::TextureOptions::LINEAR),
                            );
                        }
                    }
                }
                UiMsg::Thumb(thumb::ThumbOutcome::Failed { key, message }) => {
                    self.thumb_pending.remove(&key);
                    self.thumb_failed.insert(key);
                    self.status = format!("缩略图失败: {message}");
                }
                UiMsg::AbCaptured { slot, frame, width, height, rgba } => {
                    let image = ColorImage::from_rgba_unmultiplied([width, height], &rgba);
                    let texture = ctx.load_texture(format!("ab-{slot}"), image, egui::TextureOptions::LINEAR);
                    self.ab_textures[slot] = Some(texture);
                    self.ab_data[slot] = Some(rgba);
                    self.ab_sizes[slot] = (width, height);
                    self.ab_frames[slot] = Some(frame);
                    self.ab_blend_dirty = true;
                }
                UiMsg::ExportSaved(path) => {
                    self.status = format!("已导出: {path}");
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
                ("analysis", "progress") => {
                    if let (Some(done), Some(total)) = (
                        event.get("done").and_then(Value::as_u64),
                        event.get("total").and_then(Value::as_u64),
                    ) {
                        self.status = format!("色彩分析… {done}/{total}");
                    }
                }
                ("analysis.run", "done") if job_id == self.analysis_job || self.analysis_job.is_some() => {
                    self.analysis_job = None;
                    if let Some(result) = event.get("result") {
                        let _ = self.msg_tx.send(UiMsg::AnalysisDone(result.clone()));
                    }
                }
                ("analysis.run", "error") => {
                    self.analysis_job = None;
                    let message = event.get("message").and_then(Value::as_str).unwrap_or("未知");
                    let kind = event.get("kind").and_then(Value::as_str).unwrap_or("");
                    let _ = self.msg_tx.send(UiMsg::AnalysisFailed(
                        if kind.contains("Cancelled") { "已取消".into() } else { message.into() },
                    ));
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
        // A/B 叠加混合（同分辨率才可混）。
        if self.ab_blend_dirty && self.ab_overlay {
            if let (Some(a), Some(b), (w, h), (w2, h2)) = (
                self.ab_data[0].as_ref(),
                self.ab_data[1].as_ref(),
                self.ab_sizes[0],
                self.ab_sizes[1],
            ) {
                if (w, h) == (w2, h2) && a.len() == b.len() {
                    let alpha = self.ab_alpha;
                    let blended: Vec<u8> = a
                        .iter()
                        .zip(b.iter())
                        .map(|(x, y)| (*x as f32 * (1.0 - alpha) + *y as f32 * alpha) as u8)
                        .collect();
                    let image = ColorImage::from_rgba_unmultiplied([w, h], &blended);
                    match self.ab_blend.as_mut() {
                        Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
                        None => {
                            self.ab_blend =
                                Some(ctx.load_texture("ab-blend", image, egui::TextureOptions::LINEAR))
                        }
                    }
                }
            }
            self.ab_blend_dirty = false;
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

/// egui 默认字体不含 CJK 字形；不装中文字体所有中文界面都是豆腐块。
/// 从系统字体目录加载（优先等线/微软雅黑的 ttc，退到黑体 ttf），注册为后备字体。
fn install_cjk_fonts(ctx: &egui::Context) {
    let candidates = [
        "C:/Windows/Fonts/Deng.ttf",       // 等线（Win10+ 默认，ttf 稳）
        "C:/Windows/Fonts/simhei.ttf",     // 黑体
        "C:/Windows/Fonts/msyh.ttc",       // 微软雅黑（ttc，ab_glyph 可能不支持，放最后）
    ];
    let mut bytes = None;
    for path in candidates {
        if let Ok(data) = std::fs::read(path) {
            bytes = Some(data);
            break;
        }
    }
    let Some(bytes) = bytes else {
        eprintln!("警告：未找到系统中文字体，界面中文将显示为方块");
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "system-cjk".into(),
        egui::FontData::from_owned(bytes).into(),
    );
    // 追加为后备：拉丁字符仍用默认字体，缺字落到 CJK。
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        if let Some(list) = fonts.families.get_mut(&family) {
            list.push("system-cjk".into());
        }
    }
    ctx.set_fonts(fonts);
}

fn find_repo_root() -> PathBuf {    if let Ok(root) = std::env::var("CADENSVIEW_ROOT") {
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
        egui::Panel::right("notes").default_size(320.0).show(ui, |ui| self.right_panel(ui));
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
        let row_height = 64.0;
        let total = self.state.shots.len();
        // 虚拟滚动：只有可见行布局/请求缩略图，几百镜头也不卡。
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show_rows(ui, row_height, total, |ui, range| {
                for row in range {
                    let Some(shot) = self.state.shots.get(row).cloned() else { continue };
                    let selected = shot.id == self.state.selected_id;
                    ui.horizontal(|ui| {
                        // 缩略图（懒生成，磁盘缓存）
                        let thumb_size = egui::vec2(96.0, 54.0);
                        let key = self.thumb_key(&shot);
                        let (thumb_rect, thumb_response) =
                            ui.allocate_exact_size(thumb_size, egui::Sense::click());
                        if thumb_response.clicked() {
                            follow = Some(shot.id.clone());
                        }
                        if let Some(texture) = self.thumbs.get(&key) {
                            let size = texture.size_vec2();
                            let fitted = fit_into(thumb_size, size);
                            let origin = egui::pos2(
                                thumb_rect.left() + (thumb_size.x - fitted.x) / 2.0,
                                thumb_rect.top() + (thumb_size.y - fitted.y) / 2.0,
                            );
                            ui.painter_at(thumb_rect).image(
                                texture.id(),
                                egui::Rect::from_min_size(origin, fitted),
                                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                egui::Color32::WHITE,
                            );
                            ui.painter_at(thumb_rect).rect_stroke(
                                thumb_rect, 1.0,
                                if selected { egui::Stroke::new(2.0, egui::Color32::LIGHT_BLUE) }
                                else { egui::Stroke::new(1.0, egui::Color32::DARK_GRAY) },
                                egui::StrokeKind::Inside,
                            );
                        } else {
                            self.request_thumb(&shot, &key);
                            ui.painter_at(thumb_rect).rect_filled(
                                thumb_rect, 1.0, egui::Color32::from_gray(38));
                            ui.painter_at(thumb_rect).text(
                                thumb_rect.center(), egui::Align2::CENTER_CENTER, "…",
                                egui::FontId::proportional(14.0), egui::Color32::GRAY);
                        }
                        // 文本
                        ui.vertical(|ui| {
                            let label = format!(
                                "#{} {}帧·{:.2}s{}",
                                shot.number + 1,
                                shot.end - shot.start,
                                shot.duration_s,
                                if shot.pending { " ·待分析" } else { "" },
                            );
                            let text = if shot.pending {
                                egui::RichText::new(label).color(egui::Color32::GRAY)
                            } else {
                                egui::RichText::new(label)
                            };
                            let response = ui.selectable_label(selected, text);
                            if response.clicked() {
                                follow = Some(shot.id.clone());
                            }
                            if !shot.note.is_empty() {
                                ui.small(egui::RichText::new(truncate(&shot.note, 40)).weak());
                            }
                        });
                    });
                }
            });
        if let Some(id) = follow {
            self.select_shot(&id);
        }
    }

    fn thumb_key(&self, shot: &model::ShotRow) -> String {
        match &self.index {
            Some(index) => format!("{}-{}-{}", index.cache_stem(), shot.start, shot.end),
            None => format!("noidx-{}-{}", shot.start, shot.end),
        }
    }

    fn request_thumb(&mut self, shot: &model::ShotRow, key: &str) {
        if self.thumb_pending.contains(key) || self.thumb_failed.contains(key) {
            return;
        }
        let Some(sender) = &self.thumb_tx else { return };
        let Some(index) = &self.index else { return };
        let (iw, ih) = index.size();
        if iw == 0 || ih == 0 {
            return;
        }
        let height = 54u32;
        let width = ((height as f64 * iw as f64 / ih as f64).round() as u32).max(2);
        let middle = (shot.start + shot.end) / 2;
        let seconds = index.seconds(middle.min(index.frames().saturating_sub(1)));
        let _ = sender.send(thumb::ThumbRequest {
            key: key.to_string(),
            video: index.path.clone(),
            seconds,
            width,
            height,
        });
        self.thumb_pending.insert(key.to_string());
    }

    fn right_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.right_tab, RightTab::Notes, "📝 笔记");
            ui.selectable_value(&mut self.right_tab, RightTab::Stats, "📊 统计");
            ui.selectable_value(&mut self.right_tab, RightTab::Colors, "🎨 色彩");
        });
        ui.separator();
        match self.right_tab {
            RightTab::Notes => self.note_panel(ui),
            RightTab::Stats => self.stats_panel(ui),
            RightTab::Colors => self.colors_panel(ui),
        }
    }

    fn stats_panel(&mut self, ui: &mut egui::Ui) {
        // 打开面板时自动拉取（有索引、有镜头、且缓存缺失或过期）。
        if !self.stats_loading && self.index.is_some() && self.state.completed_count > 0
            && (self.stats.is_none() || self.stats_stale)
        {
            self.stats_loading = true;
            self.call_stats();
        }
        let Some(stats) = &self.stats else {
            ui.label("先完成切镜检测，统计在此显示。");
            return;
        };
        ui.heading("剪辑统计");
        egui::Grid::new("stats-overview").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
            let rows: [(&str, Option<f64>); 5] = [
                ("镜头数", stats.overview.shot_count.map(|v| v as f64)),
                ("ASL 平均镜头时长(秒)", stats.overview.asl_s),
                ("MSL 中位数(秒)", stats.overview.msl_s),
                ("标准差(秒)", stats.overview.stddev_s),
                ("变异系数 CV", stats.overview.cv),
            ];
            for (label, value) in rows {
                ui.strong(label);
                ui.label(match value {
                    Some(v) => format!("{v:.2}"),
                    None => "—".into(),
                });
                ui.end_row();
            }
            let duration = stats.overview.scope_duration_s;
            ui.strong("总时长(秒)");
            ui.label(duration.map(|v| format!("{v:.1}")).unwrap_or("—".into()));
            ui.end_row();
            let p = &stats.distribution.percentiles_s;
            for (label, value) in [("P10", p.p10), ("P25", p.p25), ("P75", p.p75), ("P90", p.p90)] {
                ui.strong(format!("{label} 时长(秒)"));
                ui.label(value.map(|v| format!("{v:.2}")).unwrap_or("—".into()));
                ui.end_row();
            }
        });
        ui.add_space(8.0);
        ui.heading("镜头时长分布");
        self.draw_histogram(ui, &stats.distribution.duration_histogram);
        ui.add_space(8.0);
        ui.heading("切镜密度(次/分钟)");
        self.draw_density(ui, &stats.editing.cut_density_series);
        if ui.button("↻ 刷新统计").clicked() {
            self.stats_loading = true;
            self.call_stats();
        }
    }

    fn call_stats(&self) {
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        std::thread::spawn(move || {
            match rpc.call("stats.run", json!({})) {
                Ok(value) => {
                    if let Ok(parsed) = serde_json::from_value::<StatsData>(value) {
                        let _ = tx.send(UiMsg::StatsReady(parsed));
                    }
                }
                Err(error) => {
                    let _ = tx.send(UiMsg::RpcError(error));
                }
            }
            wake();
        });
    }

    /// 手绘直方图（painter 矩形），零外部依赖。
    fn draw_histogram(&self, ui: &mut egui::Ui, bins: &[model::HistBin]) {
        let height = 110.0;
        let (rect, _response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), height),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, egui::Color32::GRAY), egui::StrokeKind::Inside);
        let Some(max_count) = bins.iter().map(|b| b.count).max().filter(|m| *m > 0) else {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "无数据",
                egui::FontId::proportional(12.0), egui::Color32::GRAY);
            return;
        };
        let inner = rect.shrink(4.0);
        let n = bins.len() as f32;
        for (index, bin) in bins.iter().enumerate() {
            let bar_w = inner.width() / n;
            let x0 = inner.left() + index as f32 * bar_w;
            let bar_h = (bin.count as f32 / max_count as f32) * (inner.height() - 12.0);
            let bar = egui::Rect::from_min_size(
                egui::pos2(x0 + 1.0, inner.bottom() - bar_h),
                egui::vec2((bar_w - 2.0).max(1.0), bar_h),
            );
            painter.rect_filled(bar, 1.0, egui::Color32::from_rgb(86, 156, 214));
            if bin.count > 0 && bar_w > 14.0 {
                painter.text(
                    egui::pos2(bar.center().x, bar.top() - 7.0),
                    egui::Align2::CENTER_CENTER,
                    bin.count,
                    egui::FontId::proportional(9.0),
                    egui::Color32::GRAY,
                );
            }
        }
        // X 轴末端标注最大时长桶
        if let Some(last) = bins.last() {
            painter.text(
                egui::pos2(inner.right(), inner.bottom() + 8.0),
                egui::Align2::RIGHT_TOP,
                format!("{:.0}s", last.end_s),
                egui::FontId::proportional(9.0),
                egui::Color32::GRAY,
            );
        }
    }

    /// 手绘密度折线（painter 线段）。
    fn draw_density(&self, ui: &mut egui::Ui, points: &[model::DensityPoint]) {
        let height = 90.0;
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), height),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, egui::Color32::GRAY), egui::StrokeKind::Inside);
        let values: Vec<(f64, f64)> = points
            .iter()
            .filter_map(|p| p.cuts_per_minute.map(|v| (p.video_time_s, v)))
            .collect();
        if values.len() < 2 {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "无数据",
                egui::FontId::proportional(12.0), egui::Color32::GRAY);
            return;
        }
        let max_t = values.last().map(|v| v.0).unwrap_or(1.0).max(0.001);
        let max_v = values.iter().map(|v| v.1).fold(0.0f64, f64::max).max(0.001);
        let inner = rect.shrink(4.0);
        inner.bottom();
        let to_pos = |t: f64, v: f64| -> egui::Pos2 {
            egui::pos2(
                inner.left() + (t / max_t) as f32 * inner.width(),
                inner.bottom() - (v / max_v) as f32 * (inner.height() - 10.0),
            )
        };
        let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(220, 120, 60));
        for pair in values.windows(2) {
            let (t0, v0) = pair[0];
            let (t1, v1) = pair[1];
            painter.line_segment([to_pos(t0, v0), to_pos(t1, v1)], stroke);
        }
        painter.text(
            egui::pos2(inner.right(), inner.bottom() + 8.0),
            egui::Align2::RIGHT_TOP,
            format!("{max_t:.0}s"),
            egui::FontId::proportional(9.0),
            egui::Color32::GRAY,
        );
        painter.text(
            egui::pos2(inner.left(), inner.top() - 2.0),
            egui::Align2::LEFT_BOTTOM,
            format!("峰值 {max_v:.1}/分"),
            egui::FontId::proportional(9.0),
            egui::Color32::GRAY,
        );
    }

    fn colors_panel(&mut self, ui: &mut egui::Ui) {
        // -- 选中镜头调色板 -----------------------------------------------------
        ui.heading("镜头调色板");
        let selected = self.state.shots.iter()
            .find(|s| s.id == self.state.selected_id)
            .cloned();
        if let Some(shot) = selected.filter(|s| !s.pending) {
            if self.palette_shot != shot.id || (self.palette.is_none() && !self.palette_loading) {
                self.palette_loading = true;
                let shot_id = shot.id.clone();
                self.palette_shot = shot_id.clone();
                self.call_palette(shot_id, (shot.start + shot.end) / 2);
            }
            ui.label(format!("镜头 #{} 中点帧的五种主色", shot.number));
            match &self.palette {
                Some(palette) if !palette.colors.is_empty() => {
                    let width = ui.available_width();
                    let swatch = (width / palette.colors.len() as f32).min(52.0);
                    ui.horizontal(|ui| {
                        for color in &palette.colors {
                            let [r, g, b] = color.srgb.unwrap_or([128, 128, 128]);
                            let (rect, _) = ui.allocate_exact_size(
                                egui::vec2(swatch, 44.0), egui::Sense::hover());
                            ui.painter_at(rect).rect_filled(
                                rect, 3.0, egui::Color32::from_rgb(r, g, b));
                            ui.painter_at(rect).rect_stroke(
                                rect, 3.0, egui::Stroke::new(1.0, egui::Color32::BLACK),
                                egui::StrokeKind::Inside);
                        }
                    });
                    ui.horizontal(|ui| {
                        for color in &palette.colors {
                            let share = color.pixel_share.unwrap_or(0.0);
                            ui.monospace(format!("{}·{}%", color.hex, (share * 100.0) as u32));
                        }
                    });
                }
                _ if self.palette_loading => { ui.spinner(); }
                _ => { ui.label("（无数据）"); }
            }
        } else {
            ui.label("选中一个已完成镜头查看调色板");
        }
        ui.add_space(10.0);
        ui.separator();
        // -- 全片色彩时间轴 -----------------------------------------------------
        ui.heading("全片色带");
        if self.barcode.is_empty() {
            if self.analysis_job.is_some() {
                ui.horizontal(|ui| { ui.spinner(); ui.label("分析进行中…"); });
            } else {
                ui.label("对整部视频按秒采样，生成时间轴色带与镜头色板。");
                let ready = self.index.is_some() && self.state.completed_count > 0;
                if ui.add_enabled(ready && !self.index_busy, egui::Button::new("▶ 运行色彩分析"))
                    .clicked()
                {
                    self.run_analysis();
                }
            }
        } else {
            let height = 26.0;
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(ui.available_width(), height), egui::Sense::hover());
            let painter = ui.painter_at(rect);
            painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, egui::Color32::GRAY), egui::StrokeKind::Inside);
            let n = self.barcode.len().max(1) as f32;
            let inner = rect.shrink(2.0);
            let slice = inner.width() / n;
            for (index, sample) in self.barcode.iter().enumerate() {
                let Some([r, g, b]) = sample.srgb else { continue };
                let cell = egui::Rect::from_min_size(
                    egui::pos2(inner.left() + index as f32 * slice, inner.top()),
                    egui::vec2(slice.max(1.0), inner.height()),
                );
                painter.rect_filled(cell, 0.0, egui::Color32::from_rgb(r, g, b));
            }
            ui.small(format!("{} 个采样点（横向 = 时间）", self.barcode.len()));
            if ui.button("↻ 重新分析").clicked() {
                self.run_analysis();
            }
        }
    }

    fn call_palette(&self, shot_id: String, frame: u64) {
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        std::thread::spawn(move || match rpc.call("palette.get", json!({"frame": frame})) {
            Ok(value) => {
                if let Ok(palette) = serde_json::from_value::<PaletteInfo>(value) {
                    let _ = tx.send(UiMsg::PaletteReady { shot_id, palette });
                }
            }
            Err(error) => {
                let _ = tx.send(UiMsg::RpcError(error));
            }
        });
        wake();
    }

    fn run_analysis(&mut self) {
        if self.analysis_job.is_some() || self.index_busy {
            return;
        }
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        self.status = "色彩分析启动…".into();
        std::thread::spawn(move || {
            match rpc.call("job.start", json!({"method": "analysis.run", "params": {}})) {
                Ok(value) => {
                    if let Some(job_id) = value.get("job_id").and_then(Value::as_u64) {
                        let _ = tx.send(UiMsg::AnalysisJobStarted(job_id));
                    }
                }

                Err(error) => {
                    let _ = tx.send(UiMsg::AnalysisFailed(error));
                }
            }
            wake();
        });
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
        // -- A/B 对照与导出 -----------------------------------------------------
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.weak("A/B:");
            if ui.button("设为 A").clicked() {
                self.capture_ab(0);
            }
            if ui.button("设为 B").clicked() {
                self.capture_ab(1);
            }
            let both = self.ab_textures[0].is_some() && self.ab_textures[1].is_some();
            if ui.add_enabled(both, egui::Button::new(if self.ab_show { "关闭对照" } else { "对照" }))
                .clicked()
            {
                self.ab_show = !self.ab_show;
            }
            ui.separator();
            if ui.button("📷 截帧 PNG").clicked() {
                self.export_frame();
            }
            if ui
                .add_enabled(self.state.completed_count > 0, egui::Button::new("📋 镜头 CSV"))
                .clicked()
            {
                self.export_shots_csv();
            }
        });
        // A/B 帧信息行
        if self.ab_textures.iter().any(|t| t.is_some()) {
            ui.horizontal(|ui| {
                for (slot, name) in [(0, "A"), (1, "B")] {
                    match self.ab_frames[slot] {
                        Some(frame) => { ui.monospace(format!("{name}: 帧 {frame}")); }
                        None => { ui.weak(format!("{name}: 未设")); }
                    }
                }
                if self.ab_show && self.ab_textures[0].is_some() && self.ab_textures[1].is_some() {
                    if ui
                        .selectable_label(!self.ab_overlay, "并排")
                        .clicked()
                    {
                        self.ab_overlay = false;
                    }
                    if ui
                        .selectable_label(self.ab_overlay, "叠加")
                        .clicked()
                    {
                        self.ab_overlay = true;
                    }
                }
            });
        }
        // A/B 显示区（替代主画面区域下方追加一块）
        if self.ab_show {
            self.ab_view(ui);
        }
    }

    fn capture_ab(&mut self, slot: usize) {
        let Some(index) = self.index.clone() else { return };
        let frame = self.player.as_ref().map(|p| p.frame()).unwrap_or(self.ui_frame);
        let ffmpeg = self.ffmpeg.clone();
        let video = PathBuf::from(&index.path);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        self.status = format!("采集 {} 帧用于对照…", if slot == 0 { "A" } else { "B" });
        std::thread::spawn(move || {
            match crate::player::capture_frame(&ffmpeg, &video, index, frame) {
                Ok((width, height, rgba)) => {
                    let _ = tx.send(UiMsg::AbCaptured { slot, frame, width, height, rgba });
                }
                Err(error) => {
                    let _ = tx.send(UiMsg::RpcError(error));
                }
            }
            wake();
        });
    }

    fn ab_view(&mut self, ui: &mut egui::Ui) {
        let (Some(a), Some(b)) = (&self.ab_textures[0], &self.ab_textures[1]) else {
            ui.label("先设好 A 和 B 两帧");
            return;
        };
        let avail = ui.available_width();
        if self.ab_overlay {
            // 叠加：CPU 混合成单图（滑条调节透明度，实时更新）。
            ui.horizontal(|ui| {
                ui.weak("B 透明度");
                let response = ui.add(
                    egui::Slider::new(&mut self.ab_alpha, 0.0..=1.0).show_value(true),
                );
                if response.changed() {
                    self.ab_blend_dirty = true;
                }
            });
            if let Some(blend) = &self.ab_blend {
                let size = blend.size_vec2();
                ui.add(
                    egui::Image::new(egui::load::SizedTexture::new(blend.id(), size))
                        .max_width(avail)
                        .maintain_aspect_ratio(true),
                );
            }
        } else {
            // 并排：左右各半。
            ui.horizontal(|ui| {
                for texture in [a, b] {
                    let size = texture.size_vec2();
                    ui.add(
                        egui::Image::new(egui::load::SizedTexture::new(texture.id(), size))
                            .max_width(avail / 2.0 - 4.0)
                            .maintain_aspect_ratio(true),
                    );
                }
            });
        }
    }

    fn export_frame(&mut self) {
        let Some(index) = &self.index else { return };
        let Some(video) = self.state.video_path.clone() else { return };
        let frame = self.ui_frame;
        let stem = Path::new(&video).file_stem().map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "frame".into());
        let default_name = format!("{stem}-帧{frame:06}.png");
        let Some(path) = rfd::FileDialog::new()
            .add_filter("PNG 图片", &["png"])
            .set_file_name(default_name)
            .save_file()
        else {
            return;
        };
        let seconds = index.seconds(frame.min(index.frames().saturating_sub(1)));
        let ffmpeg = self.ffmpeg.clone();
        let video = PathBuf::from(video);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        self.status = "导出截帧…".into();
        std::thread::spawn(move || {
            let output = std::process::Command::new(&ffmpeg)
                .arg("-nostdin").arg("-loglevel").arg("error")
                .arg("-ss").arg(format!("{seconds:.6}"))
                .arg("-i").arg(&video)
                .args(["-map", "0:v:0", "-an", "-frames:v", "1"])
                .arg(&path)
                .stdin(std::process::Stdio::null())
                .output();
            let message = match output {
                Ok(out) if out.status.success() => UiMsg::ExportSaved(path.to_string_lossy().into_owned()),
                Ok(out) => UiMsg::RpcError(format!("ffmpeg 截帧失败: {}", String::from_utf8_lossy(&out.stderr))),
                Err(error) => UiMsg::RpcError(format!("ffmpeg 启动失败: {error}")),
            };
            let _ = tx.send(message);
            wake();
        });
    }

    fn export_shots_csv(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("CSV 表格", &["csv"])
            .set_file_name("镜头汇总.csv")
            .save_file()
        else {
            return;
        };
        self.call_quiet_result("export.shots_csv", json!({"path": path.to_string_lossy()}));
    }

    fn call_quiet_result(&self, method: &str, params: Value) {
        let method = method.to_string();
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        std::thread::spawn(move || {
            match rpc.call(&method, params) {
                Ok(value) => {
                    if let Some(path) = value.get("path").and_then(Value::as_str) {
                        let _ = tx.send(UiMsg::ExportSaved(path.to_string()));
                    }
                }
                Err(error) => {
                    let _ = tx.send(UiMsg::RpcError(error));
                }
            }
            wake();
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

/// 等比缩放：把 source 尺寸装进 target 框（居中显示用）。
fn fit_into(target: egui::Vec2, source: egui::Vec2) -> egui::Vec2 {
    if source.x <= 0.0 || source.y <= 0.0 {
        return target;
    }
    let scale = (target.x / source.x).min(target.y / source.y);
    egui::vec2(source.x * scale, source.y * scale)
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
