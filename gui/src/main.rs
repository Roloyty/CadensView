//! CadensView 二开 GUI：egui 前端 + FFmpeg 播放 + shotlab Python 边车。
//! 所有 RPC 调用都在工作线程执行，UI 线程只消费 UiMsg。

mod audio;
mod detect;
mod model;
mod player;
mod rpc;
mod theme;
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
    ScaleJobStarted(u64),
    Thumb(thumb::ThumbOutcome),
    AbCaptured { slot: usize, frame: u64, width: usize, height: usize, rgba: Vec<u8> },
    ExportSaved(String),
    SubtitleText(Option<String>),
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
    tag_input: String,
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
    scale_job: Option<u64>,
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
    // -- 局部循环与字幕 ----------------------------------------------------------
    loop_active: bool,
    subtitle_loaded: bool,
    subtitle_text: Option<String>,
    // -- 缩放平移 --------------------------------------------------------------
    zoom: f32,
    pan: egui::Vec2,
    // -- 音频 ----------------------------------------------------------------
    audio: Option<audio::AudioHandle>,
    volume: f32,
    muted: bool,
    last_frame: u64,
    // -- 原版对齐：视图模式、帧轨、标记、跳转 --------------------------------
    video_only: bool,
    fullscreen: bool,
    list_open: bool,
    info_open: bool,
    marks: HashSet<String>,
    mark_anchor: Option<usize>,
    track_zoom: f64,
    track_start: f64,
    jump_open: bool,
    jump_time_mode: bool,
    jump_text: String,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Result<Self, String> {
        theme::install(&cc.egui_ctx);
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
            tag_input: String::new(),
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
            scale_job: None,
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
            loop_active: false,
            subtitle_loaded: false,
            subtitle_text: None,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            audio: audio::AudioHandle::spawn(),
            volume: 0.8,
            muted: false,
            last_frame: 0,
            video_only: false,
            fullscreen: false,
            list_open: true,
            info_open: true,
            marks: HashSet::new(),
            mark_anchor: None,
            track_zoom: 1.0,
            track_start: 0.0,
            jump_open: false,
            jump_time_mode: false,
            jump_text: String::new(),
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
        // 有字幕时同步刷新当前帧台词（静音分析也要看台词）
        if self.subtitle_loaded {
            if let Some(index) = &self.index {
                let us = (index.seconds(frame) * 1_000_000.0) as i64;
                self.call_subtitle(us);
            }
        }
    }

    /// 起播音频（与视频同一时间点）。无音轨时静默跳过。
    fn start_audio(&self, frame: u64) {
        let (Some(audio), Some(index), Some(video)) =
            (&self.audio, &self.index, self.state.video_path.as_ref())
        else {
            return;
        };
        if index.metadata.audio_tracks.unwrap_or(0) == 0 {
            return;
        }
        audio.set_volume(self.volume);
        audio.set_muted(self.muted);
        audio.play(Path::new(video), index.seconds(frame.min(index.frames().saturating_sub(1))));
    }

    fn start_audio_loop(&self, start: u64, end: u64) {
        let (Some(audio), Some(index), Some(video)) =
            (&self.audio, &self.index, self.state.video_path.as_ref())
        else {
            return;
        };
        if index.metadata.audio_tracks.unwrap_or(0) == 0 {
            return;
        }
        let start_s = index.seconds(start.min(index.frames().saturating_sub(1)));
        let end_s = index.seconds(end.min(index.frames().saturating_sub(1)));
        audio.set_volume(self.volume);
        audio.set_muted(self.muted);
        audio.play_loop(Path::new(video), start_s, (end_s - start_s).max(0.05));
    }

    fn call_subtitle(&self, timestamp_us: i64) {
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        std::thread::spawn(move || {
            if let Ok(value) = rpc.call("subtitles.at", json!({"timestamp_us": timestamp_us})) {
                let text = value.get("text").and_then(Value::as_str).map(str::to_string);
                let _ = tx.send(UiMsg::SubtitleText(text));
            }
            wake();
        });
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
                UiMsg::ScaleJobStarted(job_id) => self.scale_job = Some(job_id),
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
                UiMsg::SubtitleText(text) => self.subtitle_text = text,
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
                ("scale", "progress") => {
                    if let (Some(done), Some(total)) = (
                        event.get("done").and_then(Value::as_u64),
                        event.get("total").and_then(Value::as_u64),
                    ) {
                        self.status = format!("景别识别… {done}/{total}");
                    }
                }
                ("scale.run", "done") if job_id == self.scale_job || self.scale_job.is_some() => {
                    self.scale_job = None;
                    self.status = "景别识别完成".into();
                    // scale.run 返回的 state 含 scale 字段；直接拉一次最新状态。
                    self.call("project.state", json!({}));
                }
                ("scale.run", "error") => {
                    self.scale_job = None;
                    let message = event.get("message").and_then(Value::as_str).unwrap_or("未知");
                    let kind = event.get("kind").and_then(Value::as_str).unwrap_or("");
                    self.status = if kind.contains("Cancelled") {
                        "景别识别已取消".into()
                    } else {
                        format!("景别识别: {message}")
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
                // 循环播放重启：帧号回跳时音频同步重起（视频流由 player 内部重启）
                if self.loop_active && frame < self.last_frame {
                    if let Some(shot) = self.state.shots.iter()
                        .find(|s| s.id == self.state.selected_id && !s.pending)
                        .cloned()
                    {
                        self.start_audio_loop(shot.start, shot.end);
                    }
                }
                self.last_frame = frame;
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
                if let Some(audio) = &self.audio {
                    audio.stop();
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
        self.handle_shortcuts(&ctx);
        self.jump_window(&ctx);

        if self.video_only {
            // 视频模式（原版 Alt+Enter）：只留画面与控制条。
            egui::CentralPanel::default().show(ui, |ui| self.player_panel(ui));
            return;
        }

        egui::Panel::top("topbar")
            .frame(egui::Frame::new().fill(theme::top()).inner_margin(egui::Margin::symmetric(10, 6)))
            .show(ui, |ui| self.topbar(ui));
        egui::Panel::bottom("statusbar").show(ui, |ui| {
            ui.horizontal(|ui| {
                if self.state.dirty {
                    ui.colored_label(theme::accent(), "● 未保存");
                }
                let status_text = egui::RichText::new(&self.status);
                let colored = if self.status.starts_with('⚠') {
                    status_text.color(theme::error())
                } else if self.status.starts_with("已导出") || self.status.ends_with("完成") {
                    status_text.color(theme::accent())
                } else {
                    status_text
                };
                ui.label(colored);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (clock, total) = self.clock_labels();
                    ui.monospace(format!("{clock} / {total}"));
                });
            });
        });
        if self.list_open {
            egui::Panel::left("shots").default_size(230.0).show(ui, |ui| self.shot_list(ui));
        }
        if self.info_open {
            egui::Panel::right("notes").default_size(320.0).show(ui, |ui| self.right_panel(ui));
        }
        egui::CentralPanel::default().show(ui, |ui| self.player_panel(ui));
    }
}

impl App {
    /// 当前时间码 "mm:ss" 与总时长（原版 clockLabel 语义）。
    fn clock_labels(&self) -> (String, String) {
        let Some(index) = &self.index else {
            return ("00:00".into(), "00:00".into());
        };
        let seconds = index.seconds(self.ui_frame.min(index.frames().saturating_sub(1)));
        let total = index
            .metadata
            .duration_us
            .map(|us| us / 1_000_000.0)
            .unwrap_or_else(|| index.seconds(index.frames().saturating_sub(1)));
        (clock_text(seconds), clock_text(total))
    }

    /// 原版快捷键表（Main.qml:681-709）。
    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        // 文本输入/弹窗聚焦时不抢键。
        let typing = ctx.memory(|m| m.focused().is_some());
        if typing {
            return;
        }
        let keys = ctx.input(|i| ShortcutKeys {
            space: i.key_pressed(egui::Key::Space),
            left: i.key_pressed(egui::Key::ArrowLeft),
            right: i.key_pressed(egui::Key::ArrowRight),
            up: i.key_pressed(egui::Key::ArrowUp),
            down: i.key_pressed(egui::Key::ArrowDown),
            l: i.key_pressed(egui::Key::L),
            b: i.key_pressed(egui::Key::B),
            i: i.key_pressed(egui::Key::I),
            m: i.key_pressed(egui::Key::M),
            f: i.key_pressed(egui::Key::F),
            g: i.key_pressed(egui::Key::G),
            plus: i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals),
            minus: i.key_pressed(egui::Key::Minus),
            k: i.key_pressed(egui::Key::K),
            j: i.key_pressed(egui::Key::J),
            z: i.key_pressed(egui::Key::Z),
            s: i.key_pressed(egui::Key::S),
            e: i.key_pressed(egui::Key::E),
            a: i.key_pressed(egui::Key::A),
            o: i.key_pressed(egui::Key::O),
            escape: i.key_pressed(egui::Key::Escape),
            f11: i.key_pressed(egui::Key::F11),
            enter: i.key_pressed(egui::Key::Enter),
            ctrl: i.modifiers.ctrl,
            shift: i.modifiers.shift,
            alt: i.modifiers.alt,
        });
        if keys.escape {
            if self.jump_open {
                self.jump_open = false;
                return;
            }
            if self.video_only {
                self.video_only = false;
                return;
            }
            if !self.marks.is_empty() {
                self.marks.clear();
                self.mark_anchor = None;
                return;
            }
        }
        if keys.f11 {
            self.fullscreen = !self.fullscreen;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.fullscreen));
            return;
        }
        if keys.alt && keys.enter {
            self.video_only = !self.video_only;
            return;
        }
        let frames = self.index.as_ref().map(|i| i.frames()).unwrap_or(0);
        if frames == 0 {
            if keys.o && keys.ctrl {
                self.pick_video();
            }
            return;
        }

        // 播放 / 步进 / 跳镜头
        if keys.space {
            self.toggle_playback();
        }
        if keys.left && !keys.ctrl {
            self.step_ui(-1);
        }
        if keys.right && !keys.ctrl {
            self.step_ui(1);
        }
        if keys.up || (keys.left && keys.ctrl) {
            self.adjacent_shot(-1);
        }
        if keys.down || (keys.right && keys.ctrl) {
            self.adjacent_shot(1);
        }
        if keys.l {
            self.toggle_loop();
        }
        if keys.b {
            self.commit_note();
            self.list_open = !self.list_open;
        }
        if keys.i && self.state.completed_count > 0 {
            self.toggle_info();
        }
        if keys.m {
            self.muted = !self.muted;
            if let Some(audio) = &self.audio {
                audio.set_muted(self.muted);
            }
        }
        if keys.plus {
            self.volume = (self.volume + 0.05).min(1.0);
            if let Some(audio) = &self.audio {
                audio.set_volume(self.volume);
            }
        }
        if keys.minus {
            self.volume = (self.volume - 0.05).max(0.0);
            if let Some(audio) = &self.audio {
                audio.set_volume(self.volume);
            }
        }
        if keys.f {
            if self.index.is_some() {
                self.jump_open = true;
                self.jump_time_mode = false;
                self.jump_text.clear();
            }
        }
        if keys.g {
            if self.index.is_some() {
                self.jump_open = true;
                self.jump_time_mode = true;
                self.jump_text.clear();
            }
        }
        // 编辑
        if keys.ctrl && keys.k {
            if self.state.selected_id.is_empty() {
                return;
            }
            let frame = self.player.as_ref().map(|p| p.frame()).unwrap_or(self.ui_frame);
            let within = self
                .state
                .shots
                .iter()
                .find(|s| s.id == self.state.selected_id)
                .map(|s| frame >= s.start && frame < s.end && !s.pending)
                .unwrap_or(false);
            if within && frame > 0 {
                self.call("edit.split", json!({"frame": frame}));
            }
        }
        if keys.ctrl && keys.j {
            if let Some(shot) = self
                .state
                .shots
                .iter()
                .find(|s| s.id == self.state.selected_id && !s.pending)
                .cloned()
            {
                if shot.number + 1 < self.state.completed_count as usize {
                    self.call("edit.merge", json!({"row": shot.number}));
                }
            }
        }
        if keys.ctrl && keys.z && keys.shift {
            if self.state.can_redo {
                self.call("edit.redo", json!({}));
            }
        } else if keys.ctrl && keys.z {
            if self.state.can_undo {
                self.call("edit.undo", json!({}));
            }
        }
        if keys.ctrl && keys.s {
            self.save_project(false);
        }
        if keys.ctrl && keys.e && keys.shift {
            if self.state.completed_count > 0 {
                self.export_contact_sheet();
            }
        } else if keys.ctrl && keys.e {
            if self.index.is_some() {
                self.export_frame();
            }
        }
        if keys.ctrl && keys.a && keys.shift {
            self.marks.clear();
            self.mark_anchor = None;
        } else if keys.ctrl && keys.a && self.list_open {
            self.marks = self.state.shots.iter().filter(|s| !s.pending).map(|s| s.id.clone()).collect();
        }
        if keys.ctrl && keys.o {
            if !self.index_busy {
                self.pick_video();
            }
        }
    }

    fn pick_video(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("视频", &["mkv", "mp4", "avi", "mov", "webm", "ts", "m2ts", "flv"])
            .pick_file()
        {
            self.open_video(path);
        }
    }

    fn toggle_info(&mut self) {
        self.commit_note();
        self.info_open = !self.info_open;
    }

    fn toggle_playback(&mut self) {
        let Some(player) = &self.player else { return };
        let playing = player.is_playing();
        if playing {
            let frame = player.frame();
            player.stop();
            player.show(frame);
            if let Some(audio) = &self.audio {
                audio.stop();
            }
            self.loop_active = false;
            self.commit_cursor(frame);
        } else {
            let frame = self.ui_frame;
            player.play(frame);
            self.start_audio(frame);
        }
    }

    fn toggle_loop(&mut self) {
        let Some(shot) = self
            .state
            .shots
            .iter()
            .find(|s| s.id == self.state.selected_id && !s.pending)
            .cloned()
        else {
            return;
        };
        if self.loop_active {
            self.loop_active = false;
            if let Some(player) = &self.player {
                player.stop();
                player.show(self.ui_frame);
            }
            if let Some(audio) = &self.audio {
                audio.stop();
            }
        } else {
            self.loop_active = true;
            if let Some(player) = &self.player {
                player.play_loop(shot.start, shot.end);
            }
            self.start_audio_loop(shot.start, shot.end);
        }
    }

    fn step_ui(&mut self, delta: i64) {
        let frames = self.index.as_ref().map(|i| i.frames()).unwrap_or(0);
        if frames == 0 {
            return;
        }
        self.step(delta, frames);
    }

    fn adjacent_shot(&mut self, delta: i64) {
        let index = self
            .state
            .shots
            .iter()
            .position(|s| s.id == self.state.selected_id);
        let Some(current) = index else { return };
        let target = (current as i64 + delta)
            .clamp(0, self.state.shots.len() as i64 - 1) as usize;
        if target == current {
            return;
        }
        let shot = self.state.shots[target].clone();
        if shot.pending {
            return;
        }
        self.select_shot(&shot.id);
    }

    /// 跳转框（原版 F 帧号 / G 时间码）。
    fn jump_window(&mut self, ctx: &egui::Context) {
        if !self.jump_open {
            return;
        }
        let mut open = true;
        let title = if self.jump_time_mode { "跳转到时间" } else { "跳转到帧" };
        let hint = if self.jump_time_mode { "mm:ss 或秒" } else { "帧号（0 起）" };
        let mut commit = false;
        egui::Window::new(title)
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                ui.label(hint);
                let response = ui.add(
                    egui::TextEdit::singleline(&mut self.jump_text).desired_width(180.0),
                );
                response.request_focus();
                let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
                if ui.button("跳转").clicked() || enter {
                    commit = true;
                }
            });
        if !open {
            self.jump_open = false;
            return;
        }
        if commit {
            if let Some(index) = &self.index {
                let frames = index.frames();
                let text = self.jump_text.trim();
                let frame = if self.jump_time_mode {
                    parse_timecode(text).map(|seconds| {
                        let target = seconds as f64;
                        // 找最接近的帧
                        let mut best = 0u64;
                        let mut best_delta = f64::MAX;
                        for (number, _) in index.pts.iter().enumerate() {
                            let delta = (index.seconds(number as u64) - target).abs();
                            if delta < best_delta {
                                best_delta = delta;
                                best = number as u64;
                            }
                        }
                        best
                    })
                } else {
                    text.parse::<u64>().ok()
                };
                if let Some(frame) = frame {
                    let frame = frame.min(frames.saturating_sub(1));
                    self.ui_frame = frame;
                    if let Some(player) = &self.player {
                        player.show(frame);
                    }
                    self.commit_cursor(frame);
                    self.jump_open = false;
                }
            }
        }
    }
}

struct ShortcutKeys {
    space: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    l: bool,
    b: bool,
    i: bool,
    m: bool,
    f: bool,
    g: bool,
    plus: bool,
    minus: bool,
    k: bool,
    j: bool,
    z: bool,
    s: bool,
    e: bool,
    a: bool,
    o: bool,
    escape: bool,
    f11: bool,
    enter: bool,
    ctrl: bool,
    shift: bool,
    alt: bool,
}


fn clock_text(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    format!("{:02}:{:02}", total / 60, total % 60)
}

/// 解析 "mm:ss" / "h:mm:ss" / 纯秒数。
fn parse_timecode(text: &str) -> Option<f64> {
    let parts: Vec<&str> = text.split(':').collect();
    match parts.len() {
        1 => parts[0].trim().parse::<f64>().ok(),
        2 => {
            let minutes = parts[0].trim().parse::<f64>().ok()?;
            let seconds = parts[1].trim().parse::<f64>().ok()?;
            Some(minutes * 60.0 + seconds)
        }
        3 => {
            let hours = parts[0].trim().parse::<f64>().ok()?;
            let minutes = parts[1].trim().parse::<f64>().ok()?;
            let seconds = parts[2].trim().parse::<f64>().ok()?;
            Some(hours * 3600.0 + minutes * 60.0 + seconds)
        }
        _ => None,
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
            // 景别识别（走边车 job，可取消）
            {
                let scale_ready = self.index.is_some() && self.state.completed_count > 0
                    && self.scale_job.is_none() && !self.index_busy;
                if self.scale_job.is_some() {
                    ui.spinner();
                    if ui.button("取消景别").clicked() {
                        if let Some(job_id) = self.scale_job {
                            self.call_quiet("job.cancel", json!({"job_id": job_id}));
                        }
                    }
                } else if ui
                    .add_enabled(scale_ready, egui::Button::new("🎯 景别识别"))
                    .clicked()
                {
                    self.run_scale();
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
                ui.colored_label(theme::warning(), "（有待分析尾段：先跑检测）");
            }
            // 右侧：主题 / 视频模式 / 全屏（原版「设置」菜单语义）
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.selectable_label(self.fullscreen, "全屏").clicked() {
                    self.fullscreen = !self.fullscreen;
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.fullscreen));
                }
                if ui.selectable_label(self.video_only, "视频模式").clicked() {
                    self.video_only = !self.video_only;
                }
                ui.menu_button(format!("🎨 {}", theme::current_title()), |ui| {
                    let current = theme::current_index();
                    ui.label(egui::RichText::new(theme::current_key()).small().weak());
                    for (index, (key, title)) in theme::THEME_KEYS.iter().enumerate() {
                        let _ = key;
                        if ui
                            .selectable_label(current == index, *title)
                            .clicked()
                        {
                            theme::set_current(index);
                            theme::install(ui.ctx());
                            ui.ctx().request_repaint();
                            ui.close();
                        }
                    }
                });
            });
        });
    }

    fn shot_list(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("镜头");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if !self.marks.is_empty() {
                    ui.colored_label(theme::accent(), format!("已标记 {}", self.marks.len()));
                }
                ui.weak(format!("{:03}", self.state.shots.len()));
            });
        });
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
                                theme::ink(),
                            );
                            ui.painter_at(thumb_rect).rect_stroke(
                                thumb_rect, 1.0,
                                if selected { egui::Stroke::new(2.0, theme::accent()) }
                                else { egui::Stroke::NONE },
                                egui::StrokeKind::Inside,
                            );
                        } else {
                            self.request_thumb(&shot, &key);
                            ui.painter_at(thumb_rect).rect_filled(
                                thumb_rect, 0.0, theme::matte());
                            ui.painter_at(thumb_rect).text(
                                thumb_rect.center(), egui::Align2::CENTER_CENTER, "…",
                                egui::FontId::proportional(14.0), theme::secondary());
                        }
                        // 文本
                        ui.vertical(|ui| {
                            let marked = self.marks.contains(&shot.id);
                            let scale_text = shot.scale.as_ref()
                                .map(|s| format!(" [{}]", s.label_zh))
                                .unwrap_or_default();
                            let label = format!(
                                "{}{} {}帧·{:.2}s{}{}",
                                if marked { "● " } else { "" },
                                shot.number + 1,
                                shot.end - shot.start,
                                shot.duration_s,
                                scale_text,
                                if shot.pending { " ·待分析" } else { "" },
                            );
                            let text = if shot.pending {
                                egui::RichText::new(label).color(theme::secondary())
                            } else {
                                egui::RichText::new(label)
                            };
                            let response = ui.selectable_label(selected, text);
                            if response.clicked() {
                                let (ctrl, shift) = ui.input(|i| (i.modifiers.ctrl, i.modifiers.shift));
                                if ctrl {
                                    if self.marks.contains(&shot.id) {
                                        self.marks.remove(&shot.id);
                                    } else {
                                        self.marks.insert(shot.id.clone());
                                    }
                                    self.mark_anchor = Some(row);
                                } else if shift {
                                    let anchor = self.mark_anchor.unwrap_or(row);
                                    let (low, high) = if anchor <= row { (anchor, row) } else { (row, anchor) };
                                    for candidate in &self.state.shots[low..=high] {
                                        if !candidate.pending {
                                            self.marks.insert(candidate.id.clone());
                                        }
                                    }
                                    self.mark_anchor = Some(row);
                                } else {
                                    follow = Some(shot.id.clone());
                                    self.marks.clear();
                                    self.mark_anchor = Some(row);
                                }
                            }
                            if !shot.note.is_empty() {
                                ui.small(egui::RichText::new(truncate(&shot.note, 40)).weak());
                            }
                            if !shot.tags.is_empty() {
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 2.0;
                                    for tag in &shot.tags {
                                        ui.label(
                                            egui::RichText::new(format!(" [{tag}]"))
                                                .small()
                                                .color(theme::accent()),
                                        );
                                    }
                                });
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
        // -- 滑动平均镜头时长（节奏趋势） -------------------------------------
        if !stats.editing.rolling_asl_series.is_empty() {
            ui.add_space(8.0);
            ui.heading("滑动平均 ASL（节奏趋势）");
            let points: Vec<(f64, f64)> = stats.editing.rolling_asl_series.iter()
                .filter_map(|p| p.asl_s.map(|v| (p.video_time_s, v)))
                .collect();
            self.draw_series(ui, &points, "秒");
        }
        // -- 最密集切镜窗口 ----------------------------------------------------
        if !stats.special.densest_cut_windows.is_empty() {
            ui.add_space(8.0);
            ui.heading("切镜最密集段落");
            for (i, window) in stats.special.densest_cut_windows.iter().take(3).enumerate() {
                let start = window.start_time_s;
                let end = window.end_time_s;
                let cpm = window.cuts_per_minute.unwrap_or(0.0);
                ui.label(format!("{}. {:.0}s – {:.0}s · {:.1} 次/分", i + 1, start, end, cpm));
            }
        }
        // -- 短镜头连击（快切检测） --------------------------------------------
        if !stats.special.short_shot_runs.is_empty() {
            ui.add_space(8.0);
            ui.heading("快切段落（连续短镜头）");
            for (i, run) in stats.special.short_shot_runs.iter().take(3).enumerate() {
                match run.start_time_s {
                    Some(start) => ui.label(format!("{}. {:.1}s 起 · {} 个短镜头", i + 1, start, run.shot_count)),
                    None => ui.label(format!("{}. {} 个短镜头", i + 1, run.shot_count)),
                };
            }
        }
        // -- 时间分段汇总 ------------------------------------------------------
        if !stats.editing.time_segment_summary.is_empty() {
            ui.add_space(8.0);
            ui.heading("分段节奏（按分钟）");
            egui::Grid::new("segments").num_columns(4).spacing([10.0, 3.0]).show(ui, |ui| {
                ui.weak("时段");
                ui.weak("镜头数");
                ui.weak("ASL");
                ui.weak("次/分");
                ui.end_row();
                for row in stats.editing.time_segment_summary.iter().take(12) {
                    ui.label(format!("{:.0}–{:.0}s", row.start_time_s, row.end_time_s));
                    ui.label(row.shot_count.to_string());
                    ui.label(row.asl_s.map(|v| format!("{v:.1}")).unwrap_or("—".into()));
                    ui.label(row.cuts_per_minute.map(|v| format!("{v:.1}")).unwrap_or("—".into()));
                    ui.end_row();
                }
            });
        }
        if ui.button("↻ 刷新统计").clicked() {
            self.stats_loading = true;
            self.call_stats();
        }
    }

    /// 通用折线绘制（滑动 ASL 等单值序列）。
    fn draw_series(&self, ui: &mut egui::Ui, points: &[(f64, f64)], unit: &str) {
        let height = 80.0;
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), height), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, theme::line()),
                            egui::StrokeKind::Inside);
        if points.len() < 2 {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "无数据",
                egui::FontId::proportional(12.0), theme::secondary());
            return;
        }
        let max_t = points.last().map(|p| p.0).unwrap_or(1.0).max(0.001);
        let max_v = points.iter().map(|p| p.1).fold(0.0f64, f64::max).max(0.001);
        let inner = rect.shrink(4.0);
        let to_pos = |t: f64, v: f64| egui::pos2(
            inner.left() + (t / max_t) as f32 * inner.width(),
            inner.bottom() - (v / max_v) as f32 * (inner.height() - 10.0),
        );
        let stroke = egui::Stroke::new(1.5, theme::accent());
        for pair in points.windows(2) {
            painter.line_segment([to_pos(pair[0].0, pair[0].1), to_pos(pair[1].0, pair[1].1)], stroke);
        }
        painter.text(egui::pos2(inner.left(), inner.top() - 2.0),
            egui::Align2::LEFT_BOTTOM, format!("峰值 {max_v:.1}{unit}"),
            egui::FontId::proportional(9.0), theme::secondary());
        painter.text(egui::pos2(inner.right(), inner.bottom() + 8.0),
            egui::Align2::RIGHT_TOP, format!("{max_t:.0}s"),
            egui::FontId::proportional(9.0), theme::secondary());
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
        painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, theme::line()), egui::StrokeKind::Inside);
        let Some(max_count) = bins.iter().map(|b| b.count).max().filter(|m| *m > 0) else {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "无数据",
                egui::FontId::proportional(12.0), theme::secondary());
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
            painter.rect_filled(bar, 1.0, theme::accent());
            if bin.count > 0 && bar_w > 14.0 {
                painter.text(
                    egui::pos2(bar.center().x, bar.top() - 7.0),
                    egui::Align2::CENTER_CENTER,
                    bin.count,
                    egui::FontId::proportional(9.0),
                    theme::secondary(),
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
                theme::secondary(),
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
        painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, theme::line()), egui::StrokeKind::Inside);
        let values: Vec<(f64, f64)> = points
            .iter()
            .filter_map(|p| p.cuts_per_minute.map(|v| (p.video_time_s, v)))
            .collect();
        if values.len() < 2 {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "无数据",
                egui::FontId::proportional(12.0), theme::secondary());
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
        let stroke = egui::Stroke::new(1.5, theme::accent());
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
            theme::secondary(),
        );
        painter.text(
            egui::pos2(inner.left(), inner.top() - 2.0),
            egui::Align2::LEFT_BOTTOM,
            format!("峰值 {max_v:.1}/分"),
            egui::FontId::proportional(9.0),
            theme::secondary(),
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
                                rect, 3.0, egui::Stroke::new(1.0, theme::matte()),
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
            painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, theme::line()), egui::StrokeKind::Inside);
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

    fn run_scale(&mut self) {
        if self.scale_job.is_some() || self.index_busy {
            return;
        }
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        self.status = "景别识别启动…".into();
        std::thread::spawn(move || {
            match rpc.call("job.start", json!({"method": "scale.run", "params": {}})) {
                Ok(value) => {
                    if let Some(job_id) = value.get("job_id").and_then(Value::as_u64) {
                        let _ = tx.send(UiMsg::ScaleJobStarted(job_id));
                    }
                }
                Err(error) => {
                    let _ = tx.send(UiMsg::RpcError(error));
                }
            }
            wake();
        });
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
                // -- 标签区 -----------------------------------------------------
                ui.add_space(4.0);
                ui.strong("标签");
                ui.horizontal_wrapped(|ui| {
                    let mut remove = None;
                    for tag in &shot.tags {
                        let chip = egui::Button::new(format!("✕ {tag}"))
                            .small()
                            .fill(theme::accent_wash());
                        if ui.add(chip).clicked() {
                            remove = Some(tag.clone());
                        }
                    }
                    if let Some(tag) = remove {
                        let mut tags: Vec<String> = shot.tags.clone();
                        tags.retain(|t| t != &tag);
                        self.call("tags.set", json!({"shot_id": shot.id, "tags": tags}));
                    }
                });
                ui.horizontal(|ui| {
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut self.tag_input)
                            .hint_text("新标签…")
                            .desired_width(120.0),
                    );
                    let submit = response.lost_focus()
                        && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if ui.button("＋").clicked() || submit {
                        let tag = self.tag_input.trim().to_string();
                        if !tag.is_empty() {
                            let mut tags = shot.tags.clone();
                            if !tags.contains(&tag) {
                                tags.push(tag);
                                tags.sort();
                                self.call("tags.set", json!({"shot_id": shot.id, "tags": tags}));
                            }
                        }
                        self.tag_input.clear();
                    }
                });
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
                // 边界微调（原版 move_boundary）：左边界 = 本镜头起始帧
                if shot.number > 0 {
                    ui.horizontal(|ui| {
                        ui.weak("左边界");
                        let step = if ui.input(|i| i.modifiers.shift) { 10 } else { 1 };
                        let left_ok = shot.start.saturating_sub(step) > self.state.shots[shot.number - 1].start;
                        if ui.add_enabled(left_ok, egui::Button::new("◀")).clicked() {
                            self.call("edit.move_boundary",
                                json!({"boundary": shot.number, "frame": shot.start.saturating_sub(step)}));
                        }
                        if ui.add_enabled(shot.end > shot.start + step, egui::Button::new("▶")).clicked() {
                            self.call("edit.move_boundary",
                                json!({"boundary": shot.number, "frame": shot.start + step}));
                        }
                        ui.weak(format!("Shift ×10 · 当前 {}", shot.start));
                    });
                }
            }
            None => {
                ui.label("（未选中镜头）");
            }
        }
    }

    fn player_panel(&mut self, ui: &mut egui::Ui) {
        let frames = self.index.as_ref().map(|i| i.frames()).unwrap_or(1).max(1);
        let reserve = 210.0;
        // 画面区：滚轮缩放（1×–16×，1.25×/步，以光标为中心）、拖拽平移、双击复位。
        let texture_id = self.texture.as_ref().map(|t| (t.id(), t.size_vec2()));
        match texture_id {
            Some((id, image_size)) => {
                let avail_w = ui.available_width().max(64.0);
                let avail_h = (ui.available_height() - reserve).max(140.0);
                let fit = (avail_w / image_size.x).min(avail_h / image_size.y);
                let display = image_size * fit;
                let (rect, response) =
                    ui.allocate_exact_size(display, egui::Sense::click_and_drag());

                if response.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll.abs() > 0.5 {
                        let steps = scroll / 120.0;
                        let next = (self.zoom * 1.25f32.powf(steps)).clamp(1.0, 16.0);
                        if let Some(pointer) = response.hover_pos() {
                            // 保持光标下的画面点不动（原版 zoomAt 语义）
                            let rel = (pointer - rect.min) / display - egui::vec2(0.5, 0.5);
                            self.pan += rel * (1.0 / self.zoom - 1.0 / next);
                        }
                        self.zoom = next;
                        clamp_pan(&mut self.pan, self.zoom);
                    }
                }
                if response.dragged() && self.zoom > 1.0 {
                    let delta = response.drag_delta();
                    self.pan.x -= delta.x * (1.0 / self.zoom) / display.x;
                    self.pan.y -= delta.y * (1.0 / self.zoom) / display.y;
                    clamp_pan(&mut self.pan, self.zoom);
                }
                if response.double_clicked() {
                    self.zoom = 1.0;
                    self.pan = egui::Vec2::ZERO;
                }

                let half = 0.5 / self.zoom;
                let center = egui::pos2(0.5 + self.pan.x, 0.5 + self.pan.y);
                let uv = egui::Rect::from_center_size(center, egui::vec2(2.0 * half, 2.0 * half));
                ui.painter_at(rect).image(id, rect, uv, egui::Color32::WHITE);
                // 字幕叠层（原版 subtitleOverlay：画面底部居中、描边）
                if let Some(text) = self.subtitle_text.as_ref().filter(|t| !t.is_empty()) {
                    let font = egui::FontId::proportional((display.y / 22.0).clamp(14.0, 26.0));
                    let anchor = egui::pos2(rect.center().x, rect.bottom() - 14.0);
                    for offset in [
                        egui::vec2(-1.5, 0.0), egui::vec2(1.5, 0.0),
                        egui::vec2(0.0, -1.5), egui::vec2(0.0, 1.5),
                        egui::vec2(1.5, 1.5), egui::vec2(-1.5, -1.5),
                    ] {
                        ui.painter_at(rect).text(
                            anchor + offset, egui::Align2::CENTER_BOTTOM, text,
                            font.clone(), egui::Color32::from_black_alpha(210));
                    }
                    ui.painter_at(rect).text(
                        anchor, egui::Align2::CENTER_BOTTOM, text, font,
                        egui::Color32::WHITE);
                }
                if self.zoom > 1.001 {
                    ui.painter_at(rect).text(
                        rect.right_top() + egui::vec2(-8.0, 6.0),
                        egui::Align2::RIGHT_TOP,
                        format!("{:.0}%", self.zoom * 100.0),
                        egui::FontId::proportional(11.0),
                        egui::Color32::from_white_alpha(200),
                    );
                }
                if self.video_only {
                    ui.painter_at(rect).text(
                        rect.left_top() + egui::vec2(8.0, 6.0),
                        egui::Align2::LEFT_TOP,
                        "Alt+Enter 退出视频模式",
                        egui::FontId::proportional(11.0),
                        egui::Color32::from_white_alpha(160),
                    );
                }
            }
            None => {
                ui.centered_and_justified(|ui| ui.colored_label(theme::faint(), "打开视频后在此播放（分析模式静音或含音轨）"));
            }
        }

        // -- 帧轨（原版 FrameTrack：细线 + 刻度 + 播放头；拖动 scrub） ----------
        self.frame_track(ui, frames);
        ui.add_space(2.0);

        // -- 传输条（原版 TransportBar：计数左 / 居中控制 / 工具右） ------------
        ui.horizontal(|ui| {
            let total = ui.available_width();
            let section = (total / 3.0).max(140.0);
            ui.allocate_ui_with_layout(
                egui::vec2(section, 26.0),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.monospace(format!("{} / {} f", self.ui_frame, frames));
                },
            );
            ui.allocate_ui_with_layout(
                egui::vec2(section, 26.0),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    let playing = self.player.as_ref().map(|p| p.is_playing()).unwrap_or(false);
                    let ready = self.player.is_some();
                    if ui.add_enabled(ready, egui::Button::new("⏮").min_size(egui::vec2(26.0, 24.0))).clicked() {
                        self.adjacent_shot(-1);
                    }
                    if ui.add_enabled(ready, egui::Button::new("◁").min_size(egui::vec2(26.0, 24.0))).clicked() {
                        self.step(-1, frames);
                    }
                    let play_label = if playing { "⏸" } else { "▶" };
                    let play_button = egui::Button::new(
                        egui::RichText::new(play_label).color(theme::on_accent()),
                    )
                    .fill(theme::accent())
                    .min_size(egui::vec2(52.0, 26.0));
                    if ui.add_enabled(ready, play_button).clicked() {
                        self.toggle_playback();
                    }
                    if ui.add_enabled(ready, egui::Button::new("▷").min_size(egui::vec2(26.0, 24.0))).clicked() {
                        self.step(1, frames);
                    }
                    if ui.add_enabled(ready, egui::Button::new("⏭").min_size(egui::vec2(26.0, 24.0))).clicked() {
                        self.adjacent_shot(1);
                    }
                },
            );
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width().max(120.0), 26.0),
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    let in_shot = self.state.shots.iter()
                        .any(|s| s.id == self.state.selected_id && !s.pending);
                    let loop_label = if self.loop_active { "🔁 循环中" } else { "🔁 循环 · L" };
                    if ui
                        .add_enabled(self.player.is_some() && in_shot, egui::Button::new(loop_label))
                        .clicked()
                    {
                        self.toggle_loop();
                    }
                    if ui.button("📷").on_hover_text("截帧 PNG · Ctrl+E").clicked() {
                        self.export_frame();
                    }
                    let both = self.ab_textures[0].is_some() && self.ab_textures[1].is_some();
                    if ui
                        .add_enabled(both, egui::Button::new(if self.ab_show { "对照中" } else { "A/B 对照" }))
                        .clicked()
                    {
                        self.ab_show = !self.ab_show;
                    }
                },
            );
        });
        // 字幕 / 音量 / 缩放信息行
        ui.horizontal(|ui| {
            if ui.button("💬 字幕").on_hover_text("加载 SRT").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("SRT 字幕", &["srt", "ass"])
                    .pick_file()
                {
                    self.call_quiet("subtitles.load", json!({"path": path.to_string_lossy()}));
                    self.subtitle_loaded = true;
                }
            }
            let has_audio = self.audio.is_some()
                && self.index.as_ref().map(|i| i.metadata.audio_tracks.unwrap_or(0) > 0).unwrap_or(false);
            if has_audio {
                let mute_label = if self.muted { "🔇" } else { "🔊" };
                if ui.button(mute_label).on_hover_text("静音 · M").clicked() {
                    self.muted = !self.muted;
                    if let Some(audio) = &self.audio {
                        audio.set_muted(self.muted);
                    }
                }
                let response = ui.add(
                    egui::Slider::new(&mut self.volume, 0.0..=1.0).show_value(false),
                );
                if response.changed() {
                    if let Some(audio) = &self.audio {
                        audio.set_volume(self.volume);
                    }
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(format!("缩放 {:.0}% · 双击复位", self.zoom * 100.0));
            });
        });

        // -- A/B 对照与导出 -----------------------------------------------------
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.weak("A/B:");
            if ui.button("设为 A").clicked() {
                self.capture_ab(0);
            }
            if ui.button("设为 B").clicked() {
                self.capture_ab(1);
            }
            let both = self.ab_textures[0].is_some() && self.ab_textures[1].is_some();
            if ui
                .add_enabled(both, egui::Button::new(if self.ab_show { "关闭对照" } else { "对照" }))
                .clicked()
            {
                self.ab_show = !self.ab_show;
            }
            ui.separator();
            let has_selection = self
                .state
                .shots
                .iter()
                .any(|s| s.id == self.state.selected_id && !s.pending);
            if ui.add_enabled(has_selection, egui::Button::new("🎬 GIF")).clicked() {
                self.export_gif();
            }
            if ui
                .add_enabled(self.state.completed_count > 0, egui::Button::new("📋 CSV"))
                .clicked()
            {
                self.export_shots_csv();
            }
            if ui
                .add_enabled(self.state.completed_count > 0, egui::Button::new("🖼 长图"))
                .clicked()
            {
                self.export_contact_sheet();
            }
            if ui
                .add_enabled(self.state.completed_count > 0, egui::Button::new("📄 PDF"))
                .clicked()
            {
                self.export_pdf_sheet();
            }
            if self.state.completed_count > 0 {
                ui.weak(format!("已标记 {} · Ctrl+A", self.marks.len()));
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
                    if ui.selectable_label(!self.ab_overlay, "并排").clicked() {
                        self.ab_overlay = false;
                    }
                    if ui.selectable_label(self.ab_overlay, "叠加").clicked() {
                        self.ab_overlay = true;
                    }
                }
            });
        }
        if self.ab_show {
            self.ab_view(ui);
        }
    }

    /// 原版 FrameTrack：2px 横线、最多 13 刻度、播放头；拖动 scrub，
    /// Ctrl+滚轮缩放，普通滚轮平移，双击复位。
    fn frame_track(&mut self, ui: &mut egui::Ui, frames: u64) {
        let height = 30.0;
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), height),
            egui::Sense::click_and_drag(),
        );
        if frames < 2 {
            return;
        }
        let painter = ui.painter_at(rect);
        let inset = 8.0;
        let span = (rect.width() - inset * 2.0).max(1.0);
        let count = frames as f64;
        let max_zoom = ((count - 1.0) / 4.0).max(1.0);
        self.track_zoom = self.track_zoom.clamp(1.0, max_zoom);
        let view_span = ((count - 1.0) / self.track_zoom).max(1.0);
        let max_start = (count - 1.0 - view_span).max(0.0);
        self.track_start = self.track_start.clamp(0.0, max_start);
        let start = self.track_start;
        let position = move |frame: f64| -> f32 {
            rect.left() + inset + ((frame - start) / view_span) as f32 * span
        };
        let frame_at = move |x: f32| -> u64 {
            let fraction = ((x - rect.left() - inset) / span).clamp(0.0, 1.0) as f64;
            let frame = start + fraction * view_span;
            (frame.round() as i64).clamp(0, frames as i64 - 1) as u64
        };
        let line_color = theme::line();
        let progress_color = theme::selected();
        let accent_color = theme::accent();
        let baseline = rect.center().y + 1.0;
        painter.line_segment(
            [egui::pos2(rect.left() + inset, baseline), egui::pos2(rect.left() + inset + span, baseline)],
            egui::Stroke::new(2.0, line_color),
        );
        let head = position(self.ui_frame as f64);
        if head > rect.left() + inset {
            painter.line_segment(
                [egui::pos2(rect.left() + inset, baseline), egui::pos2(head, baseline)],
                egui::Stroke::new(2.0, progress_color.gamma_multiply(0.6)),
            );
        }
        let tick_count = (view_span.floor() as i64 + 1).clamp(2, 13) as usize;
        for index in 0..tick_count {
            let fraction = index as f64 / (tick_count - 1).max(1) as f64;
            let frame = start + fraction * view_span;
            let x = position(frame);
            let long = index == 0 || index == tick_count - 1;
            let color = if self.loop_active && (frame.round() as i64 == 0 || (frame.round() as i64) + 1 == frames as i64) {
                accent_color
            } else {
                line_color
            };
            painter.line_segment(
                [egui::pos2(x, baseline - if long { 5.0 } else { 3.0 }), egui::pos2(x, baseline - 11.0)],
                egui::Stroke::new(1.0, color),
            );
        }
        painter.line_segment(
            [egui::pos2(head, baseline - 9.0), egui::pos2(head, baseline + 9.0)],
            egui::Stroke::new(2.0, progress_color),
        );
        painter.rect_filled(
            egui::Rect::from_min_size(egui::pos2(head - 3.0, baseline - 12.0), egui::vec2(6.0, 3.0)),
            1.0,
            progress_color,
        );
        if response.dragged() || response.clicked() {
            if let Some(pointer) = response.interact_pointer_pos() {
                let frame = frame_at(pointer.x);
                if frame != self.ui_frame {
                    self.ui_frame = frame;
                    if let Some(player) = &self.player {
                        player.show(frame);
                    }
                }
            }
        }
        if response.drag_stopped() || response.clicked() {
            self.commit_cursor(self.ui_frame);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.5 {
                let ctrl = ui.input(|i| i.modifiers.ctrl);
                if ctrl {
                    let steps = scroll / 120.0;
                    let anchor_fraction = response
                        .hover_pos()
                        .map(|pointer| ((pointer.x - rect.left() - inset) / span).clamp(0.0, 1.0) as f64)
                        .unwrap_or(0.5);
                    let anchor = self.track_start + anchor_fraction * view_span;
                    self.track_zoom = (self.track_zoom * 1.25f64.powf(steps as f64)).clamp(1.0, max_zoom);
                    let new_span = ((count - 1.0) / self.track_zoom).max(1.0);
                    self.track_start =
                        (anchor - anchor_fraction * new_span).clamp(0.0, (count - 1.0 - new_span).max(0.0));
                } else if self.track_zoom > 1.0 {
                    self.track_start = (self.track_start
                        - (scroll as f64 / 120.0) * view_span * 0.15)
                        .clamp(0.0, max_start);
                }
            }
            if let Some(pointer) = response.hover_pos() {
                let frame = frame_at(pointer.x);
                response.clone().on_hover_text(format!(
                    "帧 {frame} · Ctrl+滚轮缩放{}",
                    if self.track_zoom > 1.0 { " · 滚轮平移" } else { "" }
                ));
            }
        }
        if response.double_clicked() {
            self.track_zoom = 1.0;
            self.track_start = 0.0;
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

    /// 选中镜头导出为高质量 GIF（ffmpeg palettegen/paletteuse 双通道）。
    fn export_gif(&mut self) {
        let Some(shot) = self
            .state
            .shots
            .iter()
            .find(|s| s.id == self.state.selected_id && !s.pending)
            .cloned()
        else {
            return;
        };
        let Some(video) = self.state.video_path.clone() else { return };
        let stem = Path::new(&video).file_stem().map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "shot".into());
        let Some(path) = rfd::FileDialog::new()
            .add_filter("GIF 动图", &["gif"])
            .set_file_name(format!("{stem}-镜头{}.gif", shot.number + 1))
            .save_file()
        else {
            return;
        };
        let rpc = Arc::clone(&self.rpc);
        let ffmpeg = self.ffmpeg.clone();
        let video_path = PathBuf::from(video);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        self.status = format!("导出镜头 #{} GIF…", shot.number + 1);
        std::thread::spawn(move || {
            // 1) 从边车拿精确时间范围（帧号→秒）
            let range = rpc.call("export.gif_range", json!({
                "start_frame": shot.start, "end_frame": shot.end}));
            let (start_s, duration_s) = match range {
                Ok(v) => (
                    v.get("start_s").and_then(Value::as_f64).unwrap_or(0.0),
                    v.get("duration_s").and_then(Value::as_f64).unwrap_or(1.0).max(0.1),
                ),
                Err(error) => {
                    let _ = tx.send(UiMsg::RpcError(error));
                    wake();
                    return;
                }
            };
            // 2) ffmpeg palettegen + paletteuse（高质量调色板 GIF）
            let output = std::process::Command::new(&ffmpeg)
                .arg("-nostdin").arg("-loglevel").arg("error")
                .arg("-ss").arg(format!("{start_s:.6}"))
                .arg("-t").arg(format!("{duration_s:.6}"))
                .arg("-i").arg(&video_path)
                .args(["-vf",
                    "fps=12,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                    "-loop", "0"])
                .arg(&path)
                .stdin(std::process::Stdio::null())
                .output();
            let message = match output {
                Ok(out) if out.status.success() && path.exists() => {
                    let size_kb = path.metadata().map(|m| m.len() / 1024).unwrap_or(0);
                    UiMsg::ExportSaved(format!("{} ({size_kb}KB)", path.to_string_lossy()))
                }
                Ok(out) => UiMsg::RpcError(format!(
                    "GIF 导出失败: {}",
                    String::from_utf8_lossy(&out.stderr).chars().take(300).collect::<String>())),
                Err(error) => UiMsg::RpcError(format!("ffmpeg 启动失败: {error}")),
            };
            let _ = tx.send(message);
            wake();
        });
    }

    /// 全镜头网格长图（contact sheet）。
    fn export_contact_sheet(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("PNG 图片", &["png"])
            .set_file_name("镜头分析长图.png")
            .save_file()
        else {
            return;
        };
        self.status = "生成分析长图…".into();
        self.call_quiet_result("export.contact_sheet", json!({"path": path.to_string_lossy()}));
    }

    /// PDF 联系表（原版 contact_sheet.PdfSheet 语义：分页、编号+时长+标签）。
    fn export_pdf_sheet(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("PDF 文档", &["pdf"])
            .set_file_name("镜头联系表.pdf")
            .save_file()
        else {
            return;
        };
        self.status = "生成 PDF 联系表…".into();
        self.call_quiet_result("export.pdf_sheet", json!({"path": path.to_string_lossy()}));
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

/// 把平移量限制在缩放后的可视范围内（uv 中心不得露出画面外）。
fn clamp_pan(pan: &mut egui::Vec2, zoom: f32) {
    let limit = (0.5 - 0.5 / zoom).max(0.0);
    pan.x = pan.x.clamp(-limit, limit);
    pan.y = pan.y.clamp(-limit, limit);
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

#[cfg(test)]
mod zoom_tests {
    use super::clamp_pan;
    use eframe::egui;

    #[test]
    fn zoom_one_forces_zero_pan() {
        let mut pan = egui::vec2(0.4, -0.3);
        clamp_pan(&mut pan, 1.0);
        assert_eq!(pan, egui::Vec2::ZERO, "1× 时不允许平移");
    }

    #[test]
    fn zoom_two_allows_quarter_shift() {
        let mut pan = egui::vec2(0.9, -0.9);
        clamp_pan(&mut pan, 2.0);
        // 2× 时可见范围 0.5，中心最多偏移 ±0.25
        assert!((pan.x - 0.25).abs() < 1e-6);
        assert!((pan.y + 0.25).abs() < 1e-6);
    }

    #[test]
    fn zoom_eight_limits() {
        let mut pan = egui::vec2(10.0, 10.0);
        clamp_pan(&mut pan, 8.0);
        assert!((pan.x - 0.4375).abs() < 1e-6, "0.5 - 0.5/8 = 0.4375");
    }
}
