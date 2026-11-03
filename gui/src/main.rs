//! CadensView 二开 GUI：egui 前端 + FFmpeg 播放 + shotlab Python 边车。
//! 所有 RPC 调用都在工作线程执行，UI 线程只消费 UiMsg。

mod analysis;
mod audio;
mod boundary;
mod detect;
mod model;
mod player;
mod research;
mod rpc;
mod sound;
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
    StatsReady { key: String, stats: StatsData },
    StatsFailed { key: String, error: String },
    PaletteReady { shot_id: String, palette: PaletteInfo },
    PaletteFailed { shot_id: String, error: String },
    AnalysisJobStarted(u64),
    AnalysisDone(Value),
    AnalysisFailed(String),
    ScaleJobStarted(u64),
    ScaleFailed(String),
    MotionJobStarted(u64),
    MotionDone(Value),
    MotionFailed(String),
    SlitJobStarted(u64),
    SlitDone(Value),
    SlitFailed(String),
    Thumb(thumb::ThumbOutcome),
    AbCaptured { slot: usize, frame: u64, width: usize, height: usize, rgba: Vec<u8> },
    ExportSaved(String),
    SubtitlesLoaded { project_id: String, cues: Vec<model::SubtitleCue> },
    Saved { state: State, note: Option<(String, String)> },
    AuxJobStarted(u64),
    AuxFailed(String),
    ResearchValue { method: String, value: Value },
}

/// 右侧面板的三个页签。
#[derive(Clone, Copy, PartialEq)]
enum RightTab {
    Notes,
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
    index_recovering: bool,
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
    jump_focus: bool,
    analysis_open: bool,
    analysis_view: analysis::Workspace,
    analysis_data: Option<model::ColorAnalysis>,
    stats_key: String,
    stats_error: Option<String>,
    analysis_key: String,
    analysis_starting: bool,
    motion_data: Option<model::MotionAnalysis>,
    motion_key: String,
    motion_job: Option<u64>,
    motion_starting: bool,
    slit_job: Option<u64>,
    slit_starting: bool,
    scale_starting: bool,
    boundary_editor: Option<boundary::BoundaryEditor>,
    subtitle_cues: Vec<model::SubtitleCue>,
    // -- 研究工作区与通用后台任务（导出/审校/重复镜头/VLM） ---------------------
    research: research::Research,
    sound: sound::Sound,
    aux_job: Option<u64>,
    aux_method: Option<String>,
    aux_label: String,
    cut_input: String,
    cut_shot: String,
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
            index_recovering: false,
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
            jump_focus: false,
            analysis_open: false,
            analysis_view: analysis::Workspace::default(),
            analysis_data: None,
            stats_key: String::new(),
            stats_error: None,
            analysis_key: String::new(),
            analysis_starting: false,
            motion_data: None,
            motion_key: String::new(),
            motion_job: None,
            motion_starting: false,
            slit_job: None,
            slit_starting: false,
            scale_starting: false,
            boundary_editor: None,
            subtitle_cues: Vec::new(),
            research: research::Research::default(),
            sound: sound::Sound::default(),
            aux_job: None,
            aux_method: None,
            aux_label: String::new(),
            cut_input: String::new(),
            cut_shot: String::new(),
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

    fn confirm_discard(&self) -> bool {
        if !self.state.dirty { return true; }
        rfd::MessageDialog::new()
            .set_title("未保存的修改")
            .set_description("当前项目有未保存修改。放弃并继续？")
            .set_buttons(rfd::MessageButtons::YesNo)
            .show() == rfd::MessageDialogResult::Yes
    }

    fn open_video(&mut self, video: PathBuf) {
        if self.work_busy() {
            self.status = "任务进行中，先取消再打开新视频".into();
            return;
        }
        if !self.confirm_discard() { return; }
        self.reset_project_view();
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
        if self.work_busy() { self.status = "任务进行中，先取消再打开项目".into(); return; }
        if !self.confirm_discard() { return; }
        self.reset_project_view();
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
        if self.work_busy() { self.status = "任务进行中，完成或取消后再保存".into(); return; }
        let (method, params) = if save_as || self.state.project_path.is_none() {
            let Some(path) = rfd::FileDialog::new()
                .add_filter("CadensView 项目", &["shotlab"])
                .set_file_name("project.shotlab").save_file() else { return; };
            let overwrite = path.exists() && rfd::MessageDialog::new()
                .set_title("替换项目文件")
                .set_description(format!("将当前项目保存为 {}，替换已有文件？", path.display()))
                .set_buttons(rfd::MessageButtons::YesNo).show() == rfd::MessageDialogResult::Yes;
            if path.exists() && !overwrite { return; }
            ("project.save_as", json!({"path": path.to_string_lossy(), "overwrite": overwrite}))
        } else { ("project.save", json!({})) };
        // 同一线程先提交输入框草稿再存盘，避免 Ctrl+S 与失焦提交乱序。
        let note = if self.note_dirty && !self.note_shot.is_empty() { Some((self.note_shot.clone(), self.note_draft.clone())) } else { None };
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        self.status = "保存项目…".into();
        std::thread::spawn(move || {
            let result = (|| {
                if let Some((shot, text)) = &note { rpc.call("edit.set_note", json!({"shot_id": shot, "text": text}))?; }
                let state = serde_json::from_value::<State>(rpc.call(method, params)?).map_err(|e| e.to_string())?;
                Ok::<_, String>(state)
            })();
            let message = match result {
                Ok(state) => UiMsg::Saved { state, note },
                Err(error) => UiMsg::RpcError(error),
            };
            let _ = tx.send(message); wake();
        });
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

    fn work_busy(&self) -> bool {
        self.index_busy || self.detection.is_some() || self.analysis_starting || self.scale_starting
            || self.analysis_job.is_some() || self.scale_job.is_some()
            || self.motion_job.is_some() || self.motion_starting
            || self.slit_job.is_some() || self.slit_starting
            || self.aux_method.is_some()
    }

    fn stop_playback(&mut self) {
        if let Some(player) = &self.player { player.stop(); }
        if let Some(audio) = &self.audio { audio.stop(); }
        if let Some(marker) = &mut self.playback_marker { marker.set(false); }
        self.loop_active = false;
    }

    /// 索引失败后边车仍持有旧会话：重新取回索引，恢复播放器与项目状态。
    fn restore_index(&mut self) {
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        std::thread::spawn(move || {
            match rpc.call("index.get", json!({})) {
                Ok(value) => {
                    if let Err(error) = finish_index(rpc, tx.clone(), value.get("index").cloned().unwrap_or(Value::Null), Some(())) {
                        let _ = tx.send(UiMsg::IndexJobFailed(error));
                    }
                }
                Err(error) => { let _ = tx.send(UiMsg::IndexJobFailed(error)); }
            }
            wake();
        });
    }

    fn reset_project_view(&mut self) {
        self.stop_playback();
        self.player = None;
        self.index = None;
        self.index_json_path = None;
        self.texture = None;
        self.pending_image = None;
        self.ui_frame = 0;
        self.stats = None; self.stats_stale = true; self.stats_loading = false;
        self.stats_key.clear(); self.stats_error = None;
        self.analysis_data = None; self.analysis_view.reset_results();
        self.motion_data = None; self.motion_key.clear();
        self.analysis_view.scope = analysis::Scope::All;
        self.barcode.clear(); self.palette = None; self.palette_shot.clear(); self.palette_loading = false;
        self.marks.clear(); self.mark_anchor = None; self.boundary_editor = None;
        self.note_draft.clear(); self.note_shot.clear(); self.note_dirty = false;
        self.subtitle_cues.clear(); self.subtitle_loaded = false; self.subtitle_text = None;
        self.research.reset(); self.sound.reset(); self.cut_shot.clear(); self.cut_input.clear();
        self.thumbs.clear(); self.thumb_pending.clear(); self.thumb_failed.clear();
        self.ab_textures = [None, None]; self.ab_data = [None, None]; self.ab_frames = [None, None];
        self.ab_blend = None; self.ab_show = false;
        self.track_start = 0.0; self.track_zoom = 1.0; self.zoom = 1.0; self.pan = egui::Vec2::ZERO;
    }

    fn refresh_subtitle(&mut self) {
        let Some(index) = &self.index else { return; };
        let us = (index.seconds(self.ui_frame) * 1_000_000.0).round() as i64;
        let text = self.subtitle_cues.iter().filter(|cue| cue.start_us <= us && us < cue.end_us)
            .map(|cue| cue.text.as_str()).collect::<Vec<_>>().join("\n");
        self.subtitle_text = if text.is_empty() { None } else { Some(text) };
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
        let end_s = index.endpoint_seconds(end.min(index.frames()));
        audio.set_volume(self.volume);
        audio.set_muted(self.muted);
        audio.play_loop(Path::new(video), start_s, (end_s - start_s).max(0.05));
    }

    fn load_subtitles(&self, path: PathBuf) {
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        let project_id = self.state.project_id.clone();
        std::thread::spawn(move || {
            let message = match rpc.call("subtitles.load", json!({"path": path.to_string_lossy()})) {
                Ok(value) => match serde_json::from_value(value.get("entries").cloned().unwrap_or(Value::Null)) {
                    Ok(cues) => UiMsg::SubtitlesLoaded { project_id, cues },
                    Err(error) => UiMsg::RpcError(format!("字幕解析失败: {error}")),
                },
                Err(error) => UiMsg::RpcError(error),
            };
            let _ = tx.send(message); wake();
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
                    let changed_project = state.project_id != self.state.project_id;
                    if changed_project {
                        self.reset_project_view();
                        self.ui_frame = state.current_frame;
                    }
                    let structure_changed = state.shots.len() != self.state.shots.len()
                        || state.shots.iter().zip(self.state.shots.iter())
                            .any(|(new, old)| new.id != old.id || new.start != old.start || new.end != old.end);
                    if structure_changed {
                        self.stats_stale = true; self.stats_loading = false; self.stats_error = None;
                        self.analysis_data = None; self.barcode.clear(); self.analysis_view.reset_results();
                        self.motion_data = None; self.motion_key.clear(); self.sound.result = None;
                        self.palette = None; self.palette_shot.clear(); self.palette_loading = false;
                        self.marks.retain(|id| state.shots.iter().any(|s| &s.id == id && !s.pending));
                    }
                    if self.boundary_editor.as_ref().map(|b| !b.valid_for(&state.project_id, &state.shots)).unwrap_or(false) {
                        self.boundary_editor = None;
                    }
                    self.state = state;
                }
                UiMsg::Saved { state, note } => {
                    if let Some((shot, text)) = note {
                        if shot == self.note_shot && text == self.note_draft { self.note_dirty = false; }
                    }
                    let _ = self.msg_tx.send(UiMsg::State(state));
                    self.status = "项目已保存".into();
                }
                UiMsg::IndexReady { index, index_json_path } => {
                    self.index_recovering = false;
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
                    if self.index_recovering {
                        // 恢复旧索引也失败：只报告一次，避免递归重试。
                        self.index_recovering = false;
                    } else if self.state.has_index {
                        self.index_recovering = true;
                        self.restore_index();
                    }
                }
                UiMsg::StatsReady { key, stats } => {
                    if key == self.stats_key {
                        self.stats = Some(stats);
                        self.stats_loading = false;
                        self.stats_stale = false;
                        self.stats_error = None;
                    }
                }
                UiMsg::StatsFailed { key, error } => {
                    if key == self.stats_key {
                        self.stats_loading = false;
                        self.stats_stale = false;
                        self.stats_error = Some(error);
                    }
                }
                UiMsg::PaletteReady { shot_id, palette } => {
                    if shot_id == self.palette_shot && shot_id == self.state.selected_id {
                        self.palette = Some(palette);
                        self.palette_loading = false;
                    }
                }
                UiMsg::PaletteFailed { shot_id, error } => {
                    if shot_id == self.palette_shot {
                        self.palette_loading = false;
                        self.status = format!("调色板: {error}");
                    }
                }
                // 终态可能先于 start 应答到达：只有仍处于“启动中”才接受 job id，
                // 否则已完成的 id 会把界面永久锁在忙碌状态。
                UiMsg::AnalysisJobStarted(job_id) if self.analysis_starting => {
                    self.analysis_job = Some(job_id); self.analysis_starting = false;
                }
                UiMsg::AnalysisJobStarted(_) => {}
                UiMsg::ScaleJobStarted(job_id) if self.scale_starting => {
                    self.scale_job = Some(job_id); self.scale_starting = false;
                }
                UiMsg::ScaleJobStarted(_) => {}
                UiMsg::ScaleFailed(error) => { self.scale_starting = false; self.status = format!("景别识别: {error}"); }
                UiMsg::MotionJobStarted(job_id) if self.motion_starting => {
                    self.motion_job = Some(job_id); self.motion_starting = false;
                }
                UiMsg::MotionJobStarted(_) => {}
                UiMsg::MotionDone(value) => {
                    self.motion_job = None; self.motion_starting = false;
                    let key = self.current_stats_key();
                    if self.motion_key != key {
                        self.motion_data = None;
                        self.status = "运动结果已过期，未应用".into();
                    } else {
                        match serde_json::from_value::<model::MotionAnalysis>(value) {
                            Ok(parsed) => {
                                self.motion_data = Some(parsed);
                                self.status = "运动分析完成".into();
                            }
                            Err(error) => self.status = format!("运动结果解析失败: {error}"),
                        }
                    }
                }
                UiMsg::MotionFailed(error) => {
                    self.motion_job = None; self.motion_starting = false;
                    self.status = format!("运动分析: {error}");
                }
                UiMsg::SlitJobStarted(job_id) if self.slit_starting => {
                    self.slit_job = Some(job_id); self.slit_starting = false;
                }
                UiMsg::SlitJobStarted(_) => {}
                UiMsg::SlitDone(value) => {
                    self.slit_job = None; self.slit_starting = false;
                    let path = value.get("path").and_then(Value::as_str).unwrap_or("");
                    if path.is_empty() { self.status = "扫描条导出完成".into(); }
                    else { self.status = format!("已导出扫描条: {path}"); }
                }
                UiMsg::SlitFailed(error) => {
                    self.slit_job = None; self.slit_starting = false;
                    self.status = format!("扫描条导出: {error}");
                }
                UiMsg::AnalysisDone(value) => {
                    self.analysis_job = None;
                    self.analysis_starting = false;
                    let key = self.analysis_key.clone();
                    if key != self.current_stats_key() {
                        // 范围或项目已经变化：迟到结果不得冒充当前选择。
                        self.stats_stale = true;
                        self.status = "色彩结果已过期，未应用".into();
                        continue; // 只丢这条结果；同批消息与后续纹理上传照常处理
                    }
                    let stats = value.get("shot_statistics").cloned()
                        .and_then(|v| serde_json::from_value::<StatsData>(v).ok());
                    match serde_json::from_value::<model::ColorAnalysis>(value) {
                        Ok(parsed) => {
                            self.barcode = parsed.video_color_timeline.movie_barcode.clone();
                            self.analysis_data = Some(parsed);
                            if let Some(stats) = stats {
                                self.stats = Some(stats);
                                self.stats_stale = false;
                                self.stats_error = None;
                                self.stats_key = key;
                            }
                            self.status = "色彩分析完成".into();
                        }
                        Err(error) => self.status = format!("色彩结果解析失败: {error}"),
                    }
                }
                UiMsg::AnalysisFailed(error) => {
                    self.analysis_job = None;
                    self.analysis_starting = false;
                    self.status = format!("色彩分析: {error}");
                }
                UiMsg::AuxJobStarted(job_id) if self.aux_method.is_some() => self.aux_job = Some(job_id),
                UiMsg::AuxJobStarted(_) => {}
                UiMsg::AuxFailed(error) => {
                    self.aux_job = None; self.aux_method = None;
                    self.status = format!("⚠ {}: {error}", self.aux_label);
                }
                UiMsg::ResearchValue { method, value } => {
                    if method == "compare.run" { self.status = "多片比较完成".into(); }
                    if method == "vlm.config.set" { self.status = "VLM 配置已保存".into(); }
                    if self.accept_value(&method, value) { self.call("project.state", json!({})); }
                }
                UiMsg::SubtitlesLoaded { project_id, cues } => {
                    if project_id == self.state.project_id {
                        self.subtitle_cues = cues;
                        self.subtitle_loaded = true;
                        self.refresh_subtitle();
                        self.status = format!("字幕已加载 · {} 条", self.subtitle_cues.len());
                    }
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
                    // 边车只在成功时替换会话；旧视频仍在，把播放器和索引接回来。
                    if self.state.has_index && !self.index_recovering {
                        self.index_recovering = true;
                        self.restore_index();
                    }
                }
                ("analysis", "progress") => {
                    if let (Some(done), Some(total)) = (
                        event.get("done").and_then(Value::as_u64),
                        event.get("total").and_then(Value::as_u64),
                    ) {
                        self.status = format!("色彩分析… {done}/{total}");
                    }
                }
                ("analysis.run", "done") if job_id == self.analysis_job || self.analysis_job.is_some() || self.analysis_starting => {
                    self.analysis_job = None;
                    self.analysis_starting = false;
                    if let Some(result) = event.get("result") {
                        let _ = self.msg_tx.send(UiMsg::AnalysisDone(result.clone()));
                    }
                }
                ("analysis.run", "error") => {
                    self.analysis_job = None;
                    self.analysis_starting = false;
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
                ("scale.run", "done") if job_id == self.scale_job || self.scale_job.is_some() || self.scale_starting => {
                    self.scale_job = None;
                    self.scale_starting = false;
                    self.status = "景别识别完成".into();
                    // scale.run 返回的 state 含 scale 字段；直接拉一次最新状态。
                    self.call("project.state", json!({}));
                }
                ("scale.run", "error") => {
                    self.scale_job = None;
                    self.scale_starting = false;
                    let message = event.get("message").and_then(Value::as_str).unwrap_or("未知");
                    let kind = event.get("kind").and_then(Value::as_str).unwrap_or("");
                    self.status = if kind.contains("Cancelled") {
                        "景别识别已取消".into()
                    } else {
                        format!("景别识别: {message}")
                    };
                }
                ("motion", "progress") => {
                    if let (Some(done), Some(total)) = (
                        event.get("done").and_then(Value::as_u64),
                        event.get("total").and_then(Value::as_u64),
                    ) {
                        self.status = format!("运动分析… {done}/{total} 帧");
                    }
                }
                ("motion.run", "done") if job_id == self.motion_job || self.motion_job.is_some() || self.motion_starting => {
                    self.motion_job = None; self.motion_starting = false;
                    if let Some(result) = event.get("result") {
                        let _ = self.msg_tx.send(UiMsg::MotionDone(result.clone()));
                    }
                }
                ("motion.run", "error") => {
                    self.motion_job = None; self.motion_starting = false;
                    let message = event.get("message").and_then(Value::as_str).unwrap_or("未知");
                    let kind = event.get("kind").and_then(Value::as_str).unwrap_or("");
                    self.status = if kind.contains("Cancelled") { "运动分析已取消".into() }
                        else { format!("运动分析: {message}") };
                }
                ("slit", "progress") => {
                    if let (Some(done), Some(total)) = (
                        event.get("done").and_then(Value::as_u64),
                        event.get("total").and_then(Value::as_u64),
                    ) {
                        self.status = format!("扫描条导出… {done}/{total} 帧");
                    }
                }
                ("slit.run", "done") if job_id == self.slit_job || self.slit_job.is_some() || self.slit_starting => {
                    self.slit_job = None; self.slit_starting = false;
                    if let Some(result) = event.get("result") {
                        let _ = self.msg_tx.send(UiMsg::SlitDone(result.clone()));
                    }
                }
                ("slit.run", "error") => {
                    self.slit_job = None; self.slit_starting = false;
                    let message = event.get("message").and_then(Value::as_str).unwrap_or("未知");
                    let kind = event.get("kind").and_then(Value::as_str).unwrap_or("");
                    self.status = if kind.contains("Cancelled") { "扫描条导出已取消".into() }
                        else { format!("扫描条导出: {message}") };
                }
                ("export" | "review" | "repeats" | "vlm", "progress") => {
                    if let (Some(done), Some(total)) = (
                        event.get("done").and_then(Value::as_u64),
                        event.get("total").and_then(Value::as_u64),
                    ) {
                        self.status = format!("{}… {done}/{total}", self.aux_label);
                    }
                }
                (method, phase @ ("done" | "error")) if self.aux_method.as_deref() == Some(method) => {
                    let method = method.to_string();
                    self.aux_job = None; self.aux_method = None;
                    if phase == "error" {
                        let message = event.get("message").and_then(Value::as_str).unwrap_or("未知");
                        let kind = event.get("kind").and_then(Value::as_str).unwrap_or("");
                        self.status = if kind.contains("Cancelled") { format!("{}已取消", self.aux_label) }
                            else { format!("⚠ {}: {message}", self.aux_label) };
                    } else {
                        let result = event.get("result").cloned().unwrap_or(Value::Null);
                        if let Some(path) = result.get("path").and_then(Value::as_str) {
                            self.status = format!("已导出 {path}");
                        } else {
                            self.status = format!("{}完成", self.aux_label);
                        }
                        if self.accept_value(&method, result) { self.call("project.state", json!({})); }
                    }
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
                self.refresh_subtitle();
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
        self.boundary_window(&ctx);

        if self.analysis_open {
            egui::Panel::top("topbar")
                .frame(egui::Frame::new().fill(theme::top()).inner_margin(egui::Margin::symmetric(10, 6)))
                .show(ui, |ui| self.topbar(ui));
            egui::Panel::bottom("statusbar").show(ui, |ui| self.status_line(ui));
            egui::CentralPanel::default().show(ui, |ui| self.analysis_workspace(ui, &ctx));
            return;
        }
        if self.video_only {
            // 视频模式（原版 Alt+Enter）：只留画面与控制条。
            egui::CentralPanel::default().show(ui, |ui| self.player_panel(ui));
            return;
        }

        egui::Panel::top("topbar")
            .frame(egui::Frame::new().fill(theme::top()).inner_margin(egui::Margin::symmetric(10, 6)))
            .show(ui, |ui| self.topbar(ui));
        egui::Panel::bottom("statusbar").show(ui, |ui| self.status_line(ui));
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
    fn status_line(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if self.state.dirty { ui.colored_label(theme::accent(), "● 未保存"); }
            let status_text = egui::RichText::new(&self.status);
            let colored = if self.status.starts_with('⚠') {
                status_text.color(theme::error())
            } else if self.status.starts_with("已导出") || self.status.ends_with("完成") {
                status_text.color(theme::accent())
            } else { status_text };
            ui.label(colored);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let (clock, total) = self.clock_labels();
                ui.monospace(format!("{clock} / {total}"));
            });
        });
    }

    fn analysis_workspace(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(index) = &self.index else {
            ui.centered_and_justified(|ui| ui.label("打开视频并完成切镜后查看分析"));
            return;
        };
        let stem = index.cache_stem();
        let title = self.state.video_path.as_deref()
            .and_then(|path| Path::new(path).file_stem()).map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "镜头观察".into());
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.analysis_view.scope, analysis::Scope::All, "全片已完成镜头");
            ui.selectable_value(&mut self.analysis_view.scope, analysis::Scope::Marked,
                format!("已标记镜头 ({})", self.marks.len()));
            ui.separator();
            ui.add_enabled(!self.state.regions.is_empty(),
                egui::Checkbox::new(&mut self.analysis_view.exclude_regions,
                    format!("排除 OP/ED 等区段 ({})", self.state.regions.len())))
                .on_disabled_hover_text("先把片头/片尾等镜头标记为区段");
            let marked: Vec<&model::ShotRow> = self.state.shots.iter()
                .filter(|s| !s.pending && self.marks.contains(&s.id)).collect();
            let span = marked.iter().map(|s| s.start).min().zip(marked.iter().map(|s| s.end).max());
            ui.add_enabled_ui(span.is_some(), |ui| {
                ui.menu_button("标记为区段 ▾", |ui| {
                    for (code, label) in model::REGION_LABELS {
                        if ui.button(format!("{label}（{code}）")).clicked() {
                            if let Some((start, end)) = span {
                                self.call("regions.add", json!({"label": code, "start_frame": start, "end_frame": end}));
                            }
                            ui.close();
                        }
                    }
                }).response.on_disabled_hover_text("先在镜头列表标记属于片头/片尾等的镜头");
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("← 返回观察").clicked() { self.toggle_analysis(); }
            });
        });
        if matches!(self.analysis_scope_ids(), Some(ids) if ids.is_empty()) {
            ui.colored_label(theme::warning(), "尚未标记镜头：在镜头列表用 Ctrl 点击单选、Shift 点击范围，或切换为全片。");
        }
        if self.analysis_data.is_some() && self.analysis_key != self.current_stats_key() {
            // 镜头色板/色差属于旧范围；全片色带仍有效，不清 barcode。
            self.analysis_data = None;
        }
        if self.motion_data.is_some() && self.motion_key != self.current_stats_key() {
            self.motion_data = None;
        }
        if self.sound.result.is_some() && self.sound.key != self.current_stats_key() {
            self.sound.result = None;
        }
        let key = self.current_stats_key();
        if self.stats_key != key {
            // 范围变化：旧请求即使稍后返回，也不能写进新范围的界面。
            self.stats_key = key;
            self.stats_loading = false;
            self.stats_stale = true;
        }
        if self.state.completed_count > 0 && self.stats_stale && !self.stats_loading
            && !matches!(self.analysis_scope_ids(), Some(ids) if ids.is_empty())
        {
            self.stats_loading = true;
            self.stats_error = None;
            self.call_stats();
        }
        if let Some(error) = &self.stats_error { ui.colored_label(theme::error(), error); }
        let busy = self.analysis_job.is_some() || self.analysis_starting;
        let motion_busy = self.motion_job.is_some() || self.motion_starting
            || self.slit_job.is_some() || self.slit_starting;
        let actions = self.analysis_view.show(ui, &title, self.stats.as_ref(), self.analysis_data.as_ref(),
            self.motion_data.as_ref(), &self.thumbs, &stem, busy, motion_busy);
        if self.analysis_view.topic == analysis::Topic::Research {
            let research_busy = self.work_busy();
            for action in self.research.show(ui, &self.state, research_busy) {
                self.handle_research(action);
            }
        }
        if self.analysis_view.topic == analysis::Topic::Sound {
            let sound_busy = self.work_busy();
            for action in self.sound.show(ui, &self.state, sound_busy) {
                self.handle_sound(action);
            }
        }
        for action in actions {
            match action {
                analysis::Action::RunColors => self.run_analysis(),
                analysis::Action::RunMotion => self.run_motion(),
                analysis::Action::ExportMotion => self.export_motion_spectrum(),
                analysis::Action::ExportSlit => self.export_slit_scan(),
                analysis::Action::ExportStills => self.export_stills(),
                analysis::Action::Thumbnail(frame) => self.request_analysis_thumb(&stem, frame),
                analysis::Action::Locate { frame, shot_id } => {
                    self.analysis_open = false;
                    if let Some(player) = &self.player { player.show(frame); }
                    self.ui_frame = frame;
                    self.commit_cursor(frame);
                    if let Some(id) = shot_id { self.select_shot(&id); }
                }
            }
        }
        let actions = self.analysis_view.details(ctx, self.stats.as_ref(), self.analysis_data.as_ref(), &self.thumbs, &stem);
        for action in actions {
            match action {
                analysis::Action::Locate { frame, shot_id } => {
                    self.analysis_open = false;
                    if let Some(player) = &self.player { player.show(frame); }
                    self.ui_frame = frame;
                    self.commit_cursor(frame);
                    if let Some(id) = shot_id { self.select_shot(&id); }
                }
                _ => {}
            }
        }
    }

    fn request_analysis_thumb(&mut self, stem: &str, frame: u64) {
        let key = analysis::thumbnail_key(stem, frame);
        if self.thumbs.contains_key(&key) || self.thumb_pending.contains(&key) || self.thumb_failed.contains(&key) { return; }
        let (Some(sender), Some(index)) = (&self.thumb_tx, &self.index) else { return; };
        let (iw, ih) = index.size();
        if iw == 0 || ih == 0 { return; }
        let width = 320u32;
        let height = ((width as f64 * ih as f64 / iw as f64).round() as u32).max(2);
        let _ = sender.send(thumb::ThumbRequest {
            key: key.clone(), video: index.path.clone(),
            seconds: index.seconds(frame.min(index.frames().saturating_sub(1))), width, height,
        });
        self.thumb_pending.insert(key);
    }

    /// 当前时间码 "mm:ss" 与总时长（原版 clockLabel 语义）。
    fn clock_labels(&self) -> (String, String) {
        let Some(index) = &self.index else {
            return ("00:00".into(), "00:00".into());
        };
        let seconds = index.seconds(self.ui_frame.min(index.frames().saturating_sub(1)));
        let total = index.endpoint_seconds(index.frames());
        (clock_text(seconds), clock_text(total))
    }

    /// 原版快捷键表（Main.qml:681-709）。组合键不与普通键互相穿透。
    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() { return; }
        let keys = ctx.input(|i| ShortcutKeys {
            space: i.key_pressed(egui::Key::Space),
            left: i.key_pressed(egui::Key::ArrowLeft),
            right: i.key_pressed(egui::Key::ArrowRight),
            up: i.key_pressed(egui::Key::ArrowUp),
            down: i.key_pressed(egui::Key::ArrowDown),
            l: i.key_pressed(egui::Key::L), b: i.key_pressed(egui::Key::B),
            i: i.key_pressed(egui::Key::I), m: i.key_pressed(egui::Key::M),
            f: i.key_pressed(egui::Key::F), g: i.key_pressed(egui::Key::G),
            plus: i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals),
            minus: i.key_pressed(egui::Key::Minus),
            k: i.key_pressed(egui::Key::K), j: i.key_pressed(egui::Key::J),
            z: i.key_pressed(egui::Key::Z), s: i.key_pressed(egui::Key::S),
            e: i.key_pressed(egui::Key::E), a: i.key_pressed(egui::Key::A),
            o: i.key_pressed(egui::Key::O), c: i.key_pressed(egui::Key::C),
            escape: i.key_pressed(egui::Key::Escape), f11: i.key_pressed(egui::Key::F11),
            enter: i.key_pressed(egui::Key::Enter),
            ctrl: i.modifiers.ctrl, shift: i.modifiers.shift, alt: i.modifiers.alt,
        });
        let plain = !keys.ctrl && !keys.alt;
        if keys.escape {
            if self.jump_open { self.jump_open = false; return; }
            // 边界框自己处理 Esc：需要把预览帧恢复原位。
            if self.boundary_editor.is_some() { return; }
            if self.analysis_open { self.analysis_open = false; return; }
            if self.video_only { self.video_only = false; return; }
            if !self.marks.is_empty() { self.marks.clear(); self.mark_anchor = None; return; }
        }
        if keys.f11 { self.toggle_analysis(); return; }
        if keys.alt && keys.enter { self.analysis_open = false; self.video_only = !self.video_only; return; }
        if keys.ctrl && keys.shift && keys.c { self.ab_show = !self.ab_show; return; }
        if self.analysis_open { return; }
        let frames = self.index.as_ref().map(|i| i.frames()).unwrap_or(0);
        if frames == 0 {
            if keys.ctrl && keys.o { self.pick_video(); }
            return;
        }
        if plain && keys.space { self.toggle_playback(); }
        // 1–5 人工景别（大特写/特写/中景/全景/远景），0 清除。
        let digit = ctx.input(|i| [egui::Key::Num0, egui::Key::Num1, egui::Key::Num2, egui::Key::Num3,
                                    egui::Key::Num4, egui::Key::Num5].iter().position(|k| i.key_pressed(*k)));
        if let (true, Some(digit)) = (plain && !keys.shift, digit) { self.set_manual_scale(digit); }
        if plain && keys.left { self.step_ui(-1); }
        if plain && keys.right { self.step_ui(1); }
        if plain && keys.up { self.adjacent_shot(-1); }
        if plain && keys.down { self.adjacent_shot(1); }
        if keys.ctrl && keys.left { self.adjacent_shot(-1); }
        if keys.ctrl && keys.right { self.adjacent_shot(1); }
        if plain && keys.l { self.toggle_loop(); }
        if plain && keys.b { self.commit_note(); self.list_open = !self.list_open; }
        if plain && keys.i && self.state.completed_count > 0 { self.toggle_info(); }
        if plain && keys.m {
            self.muted = !self.muted;
            if let Some(audio) = &self.audio { audio.set_muted(self.muted); }
        }
        if plain && keys.plus {
            self.volume = (self.volume + 0.05).min(1.0);
            if let Some(audio) = &self.audio { audio.set_volume(self.volume); }
        }
        if plain && keys.minus {
            self.volume = (self.volume - 0.05).max(0.0);
            if let Some(audio) = &self.audio { audio.set_volume(self.volume); }
        }
        if plain && keys.f && self.index.is_some() {
            self.jump_open = true; self.jump_time_mode = false; self.jump_text.clear(); self.jump_focus = true;
        }
        if plain && keys.g && self.index.is_some() {
            self.jump_open = true; self.jump_time_mode = true; self.jump_text.clear(); self.jump_focus = true;
        }
        if keys.ctrl && keys.k {
            let frame = self.player.as_ref().map(|p| p.frame()).unwrap_or(self.ui_frame);
            let within = self.state.shots.iter().find(|s| s.id == self.state.selected_id)
                .map(|s| frame >= s.start && frame < s.end && !s.pending).unwrap_or(false);
            if within && frame > 0 { self.commit_note(); self.call("edit.split", json!({"frame": frame})); }
        }
        if keys.ctrl && keys.j {
            if let Some(shot) = self.state.shots.iter()
                .find(|s| s.id == self.state.selected_id && !s.pending).cloned()
            {
                if shot.number + 1 < self.state.completed_count as usize {
                    self.commit_note();
                    self.call("edit.merge", json!({"row": shot.number}));
                }
            }
        }
        if keys.ctrl && keys.z && keys.shift {
            if self.state.can_redo { self.call("edit.redo", json!({})); }
        } else if keys.ctrl && keys.z {
            if self.state.can_undo { self.call("edit.undo", json!({})); }
        }
        if keys.ctrl && keys.s { self.save_project(false); }
        if keys.ctrl && keys.e && keys.shift {
            if self.state.completed_count > 0 { self.export_contact_sheet(); }
        } else if keys.ctrl && keys.e {
            if self.index.is_some() { self.export_frame(); }
        }
        if keys.ctrl && keys.a && keys.shift {
            self.marks.clear(); self.mark_anchor = None;
        } else if keys.ctrl && keys.a && self.list_open {
            self.marks = self.state.shots.iter().filter(|s| !s.pending).map(|s| s.id.clone()).collect();
        }
        if keys.ctrl && keys.o && !self.work_busy() { self.pick_video(); }
    }

    fn boundary_window(&mut self, ctx: &egui::Context) {
        let action = self.boundary_editor.as_mut().and_then(|editor| editor.show(ctx));
        match action {
            Some(boundary::Action::Preview(frame)) => {
                if let Some(player) = &self.player { player.show(frame); }
                self.ui_frame = frame;
            }
            Some(boundary::Action::Apply { boundary, frame }) => {
                self.boundary_editor = None;
                self.commit_note();
                self.call("edit.move_boundary", json!({"boundary": boundary, "frame": frame}));
            }
            Some(boundary::Action::Cancel(frame)) => {
                self.boundary_editor = None;
                if let Some(player) = &self.player { player.show(frame); }
                self.ui_frame = frame;
            }
            None => {}
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

    fn toggle_analysis(&mut self) {
        self.analysis_open = !self.analysis_open;
        if self.analysis_open {
            self.stop_playback();
            self.jump_open = false;
            self.boundary_editor = None;
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
                if self.jump_focus { response.request_focus(); self.jump_focus = false; }
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
                    parse_timecode(text).map(|seconds| index.frame_at(seconds))
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
    c: bool,
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
            if self.motion_job.is_some() || self.motion_starting {
                ui.spinner();
                if let Some(job_id) = self.motion_job {
                    if ui.button("取消运动").clicked() {
                        self.call_quiet("job.cancel", json!({"job_id": job_id}));
                    }
                }
            }
            if self.slit_job.is_some() || self.slit_starting {
                ui.spinner();
                if let Some(job_id) = self.slit_job {
                    if ui.button("取消扫描条").clicked() {
                        self.call_quiet("job.cancel", json!({"job_id": job_id}));
                    }
                }
            }
            if self.aux_method.is_some() {
                ui.spinner();
                if let Some(job_id) = self.aux_job {
                    if ui.button(format!("取消{}", self.aux_label)).clicked() {
                        self.call_quiet("job.cancel", json!({"job_id": job_id}));
                    }
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
                if ui.selectable_label(self.analysis_open, "📊 分析").on_hover_text("分析工作区 · F11").clicked() {
                    self.toggle_analysis();
                }
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
                            let (ctrl, shift) = ui.input(|i| (i.modifiers.ctrl, i.modifiers.shift));
                            if ctrl {
                                if self.marks.contains(&shot.id) { self.marks.remove(&shot.id); }
                                else { self.marks.insert(shot.id.clone()); }
                                self.mark_anchor = Some(row);
                            } else if shift {
                                let anchor = self.mark_anchor.unwrap_or(row);
                                let (low, high) = if anchor <= row { (anchor, row) } else { (row, anchor) };
                                for candidate in &self.state.shots[low..=high] {
                                    if !candidate.pending { self.marks.insert(candidate.id.clone()); }
                                }
                                self.mark_anchor = Some(row);
                            } else {
                                follow = Some(shot.id.clone());
                                self.marks.clear();
                                self.mark_anchor = Some(row);
                            }
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
                            // 人工景别优先（✎），否则显示模型结果。
                            let scale_text = match (&shot.scale_manual, &shot.scale) {
                                (Some(code), _) => format!(" [✎{}]", model::scale_zh(code)),
                                (None, Some(s)) => format!(" [{}]", s.label_zh),
                                _ => String::new(),
                            };
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
            ui.selectable_value(&mut self.right_tab, RightTab::Notes, "📝 镜头");
            ui.selectable_value(&mut self.right_tab, RightTab::Colors, "🎨 色彩");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("分析工作区").on_hover_text("剪辑 / 节奏 / 色彩图表").clicked() {
                    self.toggle_analysis();
                }
            });
        });
        ui.separator();
        match self.right_tab {
            RightTab::Notes => self.note_panel(ui),
            RightTab::Colors => self.colors_panel(ui),
        }
    }

    /// None = 全部已完成镜头；Some = 用户显式标记的稳定 ID。
    fn analysis_scope_ids(&self) -> Option<Vec<String>> {
        match self.analysis_view.scope {
            analysis::Scope::All => None,
            analysis::Scope::Marked => Some(self.state.shots.iter()
                .filter(|s| !s.pending && self.marks.contains(&s.id))
                .map(|s| s.id.clone()).collect()),
        }
    }

    fn scope_params(&self) -> Result<Value, String> {
        let mut params = match self.analysis_scope_ids() {
            None => json!({}),
            Some(ids) if ids.is_empty() => return Err("先标记至少一个已完成镜头，或切换到全片范围".into()),
            Some(ids) => json!({"shot_ids": ids}),
        };
        if self.excluding_regions() { params["exclude_regions"] = json!(true); }
        Ok(params)
    }

    /// 只有项目确实有区段时排除才生效，键与参数保持一致。
    fn excluding_regions(&self) -> bool {
        self.analysis_view.exclude_regions && !self.state.regions.is_empty()
    }

    fn current_stats_key(&self) -> String {
        let regions = if self.excluding_regions() {
            format!("|x:{}", self.state.regions.iter().map(|r| format!("{}-{}", r.start_frame, r.end_frame))
                .collect::<Vec<_>>().join(","))
        } else { String::new() };
        match self.analysis_scope_ids() {
            None => format!("{}|all{regions}", self.state.project_id),
            Some(ids) => format!("{}|{}{regions}", self.state.project_id, ids.join(",")),
        }
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

    fn call_stats(&self) {
        let key = self.current_stats_key();
        let Ok(params) = self.scope_params() else { return; };
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        std::thread::spawn(move || {
            let message = match rpc.call("stats.run", params) {
                Ok(value) => match serde_json::from_value::<StatsData>(value) {
                    Ok(stats) => UiMsg::StatsReady { key, stats },
                    Err(error) => UiMsg::StatsFailed { key, error: format!("统计结果解析失败: {error}") },
                },
                Err(error) => UiMsg::StatsFailed { key, error },
            };
            let _ = tx.send(message); wake();
        });
    }

    fn call_palette(&self, shot_id: String, frame: u64) {
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        std::thread::spawn(move || {
            let message = match rpc.call("palette.get", json!({"frame": frame})) {
                Ok(value) => match serde_json::from_value::<PaletteInfo>(value) {
                    Ok(palette) => UiMsg::PaletteReady { shot_id, palette },
                    Err(error) => UiMsg::PaletteFailed { shot_id, error: format!("调色板解析失败: {error}") },
                },
                Err(error) => UiMsg::PaletteFailed { shot_id, error },
            };
            let _ = tx.send(message); wake();
        });
    }

    fn run_scale(&mut self) {
        if self.work_busy() {
            return;
        }
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        self.status = "景别识别启动…".into();
        self.scale_starting = true;
        std::thread::spawn(move || {
            match rpc.call("job.start", json!({"method": "scale.run", "params": {}})) {
                Ok(value) => {
                    if let Some(job_id) = value.get("job_id").and_then(Value::as_u64) {
                        let _ = tx.send(UiMsg::ScaleJobStarted(job_id));
                    }
                }
                Err(error) => {
                    let _ = tx.send(UiMsg::ScaleFailed(error));
                }
            }
            wake();
        });
    }

    fn run_analysis(&mut self) {
        if self.work_busy() {
            return;
        }
        let params = match self.scope_params() {
            Ok(params) => params,
            Err(error) => { self.status = error; return; }
        };
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        self.status = "色彩分析启动…".into();
        self.analysis_starting = true;
        self.analysis_key = self.current_stats_key();
        std::thread::spawn(move || {
            match rpc.call("job.start", json!({"method": "analysis.run", "params": params})) {
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

    fn run_motion(&mut self) {
        if self.work_busy() { return; }
        let params = match self.scope_params() {
            Ok(params) => params,
            Err(error) => { self.status = error; return; }
        };
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        self.status = "运动分析启动…".into();
        self.motion_starting = true;
        self.motion_key = self.current_stats_key();
        std::thread::spawn(move || {
            match rpc.call("job.start", json!({"method": "motion.run", "params": params})) {
                Ok(value) => {
                    if let Some(job_id) = value.get("job_id").and_then(Value::as_u64) {
                        let _ = tx.send(UiMsg::MotionJobStarted(job_id));
                    }
                }
                Err(error) => { let _ = tx.send(UiMsg::MotionFailed(error)); }
            }
            wake();
        });
    }

    fn export_motion_spectrum(&mut self) {
        let Some(motion) = &self.motion_data else {
            self.status = "先运行运动分析再导出指纹".into();
            return;
        };
        if self.motion_key != self.current_stats_key() {
            self.status = "范围已变化，请重新运行运动分析".into();
            return;
        }
        let duration = motion.duration_s;
        let Some(path) = rfd::FileDialog::new()
            .add_filter("PNG 图片", &["png"])
            .set_file_name(format!("运动指纹-{:.0}s.png", duration))
            .save_file() else { return; };
        match self.scope_params() {
            Ok(mut params) => {
                params["path"] = json!(path.to_string_lossy());
                self.call_quiet_result("export.motion_spectrum", params);
            }
            Err(error) => self.status = error,
        }
    }

    fn export_slit_scan(&mut self) {
        if self.work_busy() { return; }
        let count = self.analysis_scope_ids().map(|ids| ids.len())
            .unwrap_or(self.state.completed_count as usize);
        if count == 0 { self.status = "没有可导出的已完成镜头".into(); return; }
        let confirmed = rfd::MessageDialog::new()
            .set_title("导出镜头扫描条")
            .set_description(format!("将对 {count} 个镜头逐帧解码，生成 slit-scan 网格图。
长片或大量镜头可能耗时数分钟，可取消。继续？"))
            .set_buttons(rfd::MessageButtons::YesNo)
            .show() == rfd::MessageDialogResult::Yes;
        if !confirmed { return; }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("PNG 图片", &["png"])
            .set_file_name("镜头扫描条.png")
            .save_file() else { return; };
        let Ok(mut params) = self.scope_params() else { return; };
        params["path"] = json!(path.to_string_lossy());
        params["width"] = json!(160);
        params["columns"] = json!(6);
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        self.status = "扫描条导出启动…".into();
        self.slit_starting = true;
        std::thread::spawn(move || {
            match rpc.call("job.start", json!({"method": "slit.run", "params": params})) {
                Ok(value) => {
                    if let Some(job_id) = value.get("job_id").and_then(Value::as_u64) {
                        let _ = tx.send(UiMsg::SlitJobStarted(job_id));
                    }
                }
                Err(error) => { let _ = tx.send(UiMsg::SlitFailed(error)); }
            }
            wake();
        });
    }

    fn export_stills(&mut self) {
        let Some(folder) = rfd::FileDialog::new().pick_folder() else { return; };
        match self.scope_params() {
            Ok(mut params) => {
                params["path"] = json!(folder.to_string_lossy());
                params["count"] = json!(100);
                self.start_aux_job("export.stills", params, "导出 100 静帧");
            }
            Err(error) => self.status = error,
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
                if !shot.pending { self.annotation_block(ui, &shot); }
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
                    .add_enabled(shot.number + 1 < self.state.completed_count as usize, egui::Button::new("与下一镜头合并"))
                    .clicked()
                {
                    self.call("edit.merge", json!({"row": shot.number}));
                }
                // 共享边界：起点属于本镜头，终点属于下一镜头；两者指向同一帧。
                ui.horizontal(|ui| {
                    if shot.number > 0 && ui.add_enabled(!shot.pending, egui::Button::new("校正起点…")).clicked() {
                        self.boundary_editor = boundary::BoundaryEditor::new(
                            &self.state.project_id, &self.state.shots, shot.number,
                            boundary::Side::Start, current);
                    }
                    if shot.number + 1 < self.state.completed_count as usize
                        && ui.add_enabled(!shot.pending, egui::Button::new("校正终点…")).clicked()
                    {
                        self.boundary_editor = boundary::BoundaryEditor::new(
                            &self.state.project_id, &self.state.shots, shot.number,
                            boundary::Side::End, current);
                    }
                });
                if shot.number > 0 {
                    ui.horizontal(|ui| {
                        ui.weak("步进");
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
                        ui.weak(format!("Shift ×10 · 共享帧 {}", shot.start));
                    });
                }
            }
            None => {
                ui.label("（未选中镜头）");
            }
        }
    }

    /// 研究标注：カット编号、人工景别（与模型/VLM 分层显示）、所属区段。
    fn annotation_block(&mut self, ui: &mut egui::Ui, shot: &model::ShotRow) {
        ui.add_space(4.0);
        if self.cut_shot != shot.id {
            self.cut_shot = shot.id.clone();
            self.cut_input = shot.cut_number.clone();
        }
        ui.horizontal(|ui| {
            ui.strong("カット");
            let response = ui.add(egui::TextEdit::singleline(&mut self.cut_input)
                .hint_text("制作编号，如 C012").desired_width(90.0));
            if response.lost_focus() && self.cut_input.trim() != shot.cut_number {
                self.call("cut.set", json!({"shot_id": shot.id, "value": self.cut_input.trim()}));
            }
            if let Some(region) = &shot.region {
                ui.colored_label(theme::warning(), format!("区段：{}", model::region_zh(region)));
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.strong("景别");
            for (index, (code, label)) in model::SCALES.iter().enumerate() {
                let active = shot.scale_manual.as_deref() == Some(*code);
                if ui.selectable_label(active, *label).on_hover_text(format!("快捷键 {}", index + 1)).clicked() {
                    let value = if active { Value::Null } else { json!(code) };
                    self.call("scale.set_manual", json!({"shot_id": shot.id, "label": value}));
                }
            }
        });
        let model_scale = shot.scale.as_ref().map(|s| format!("模型 {}（{:.0}%）", s.label_zh, s.confidence.unwrap_or(0.0) * 100.0));
        let vlm_scale = shot.vlm.as_ref().and_then(|v| v.shot_scale.as_deref()).map(|s| format!("VLM {}", model::scale_zh(s)));
        let hints: Vec<String> = model_scale.into_iter().chain(vlm_scale).collect();
        if !hints.is_empty() { ui.weak(hints.join(" · ")); }
        if let Some(vlm) = &shot.vlm {
            ui.small(egui::RichText::new(format!("VLM：{} · {} · {}{}",
                vlm.camera_movement.as_deref().unwrap_or("—"), vlm.content_type.as_deref().unwrap_or("—"),
                vlm.description, if vlm.effects.is_empty() { String::new() } else { format!(" · {}", vlm.effects.join("/")) }))
                .color(theme::secondary()));
        }
    }

    fn set_manual_scale(&mut self, digit: usize) {
        let Some(shot) = self.state.shots.iter().find(|s| s.id == self.state.selected_id && !s.pending) else { return; };
        let label = if digit == 0 { Value::Null } else { json!(model::SCALES[digit - 1].0) };
        self.call("scale.set_manual", json!({"shot_id": shot.id, "label": label}));
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
            if ui.button("💬 字幕").on_hover_text("加载 SRT 字幕").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("SRT 字幕", &["srt"])
                    .pick_file()
                {
                    self.load_subtitles(path);
                }
            }
            if self.subtitle_loaded {
                ui.weak(format!("{} 条", self.subtitle_cues.len()));
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
                if let Some(ids) = self.analysis_scope_ids() {
                    ui.colored_label(theme::accent(), format!("导出/分析范围：已标记 {} 个", ids.len()));
                }
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
        match self.scope_params() {
            Ok(mut params) => { params["path"] = json!(path.to_string_lossy());
                self.call_quiet_result("export.shots_csv", params); }
            Err(error) => self.status = error,
        }
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
        match self.scope_params() {
            Ok(mut params) => { params["path"] = json!(path.to_string_lossy());
                self.start_aux_job("export.contact_sheet", params, "生成分析长图"); }
            Err(error) => self.status = error,
        }
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
        match self.scope_params() {
            Ok(mut params) => { params["path"] = json!(path.to_string_lossy());
                self.start_aux_job("export.pdf_sheet", params, "生成 PDF 联系表"); }
            Err(error) => self.status = error,
        }
    }

    /// 导出/审校/重复镜头/VLM 共用一个后台任务槽（边车同一时刻只跑一个任务）。
    fn start_aux_job(&mut self, method: &str, params: Value, label: &str) {
        if self.work_busy() {
            self.status = "已有任务进行中，完成或取消后再试".into();
            return;
        }
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        self.aux_method = Some(method.to_string());
        self.aux_label = label.to_string();
        self.aux_job = None;
        self.status = format!("{label}…");
        let method = method.to_string();
        std::thread::spawn(move || {
            let message = match rpc.call("job.start", json!({"method": method, "params": params})) {
                Ok(value) => match value.get("job_id").and_then(Value::as_u64) {
                    Some(job_id) => UiMsg::AuxJobStarted(job_id),
                    None => UiMsg::AuxFailed("任务没有返回 job_id".into()),
                },
                Err(error) => UiMsg::AuxFailed(error),
            };
            let _ = tx.send(message); wake();
        });
    }

    /// 结果不是 State 的同步调用：原样交给研究面板。
    fn call_value(&self, method: &str, params: Value) {
        let method = method.to_string();
        let rpc = Arc::clone(&self.rpc);
        let tx = self.msg_tx.clone();
        let wake = Arc::clone(&self.wake);
        std::thread::spawn(move || {
            let message = match rpc.call(&method, params) {
                Ok(value) => UiMsg::ResearchValue { method, value },
                Err(error) => UiMsg::RpcError(error),
            };
            let _ = tx.send(message); wake();
        });
    }

    fn locate_frame(&mut self, frame: u64) {
        self.analysis_open = false;
        if let Some(player) = &self.player { player.show(frame); }
        self.ui_frame = frame;
        if let Some(shot) = self.state.shots.iter().find(|s| s.start <= frame && frame < s.end).map(|s| s.id.clone()) {
            self.call("edit.select", json!({"shot_id": shot, "frame": frame}));
        } else {
            self.commit_cursor(frame);
        }
    }

    /// 边车结果分给声音面板或研究面板；返回 true 时需要刷新项目状态。
    fn accept_value(&mut self, method: &str, value: Value) -> bool {
        if method.starts_with("audio.") {
            return self.sound.accept(method, value).unwrap_or(false);
        }
        self.research.accept(method, value)
    }

    fn handle_sound(&mut self, action: sound::Action) {
        match action {
            sound::Action::Run => match self.scope_params() {
                Ok(params) => {
                    self.sound.key = self.current_stats_key();
                    self.start_aux_job("audio.run", params, "声音分析");
                }
                Err(error) => self.status = error,
            },
            sound::Action::Match => {
                let Some(paths) = rfd::FileDialog::new().add_filter("CadensView 项目", &["shotlab"]).pick_files() else { return; };
                let paths: Vec<String> = paths.iter().map(|p| p.to_string_lossy().into_owned()).collect();
                self.start_aux_job("audio.match", json!({"paths": paths}), "跨集比对音频");
            }
            sound::Action::AddRegion { label, start_frame, end_frame } => {
                self.call("regions.add", json!({"label": label, "start_frame": start_frame, "end_frame": end_frame}));
            }
            sound::Action::Locate(frame) => self.locate_frame(frame),
        }
    }

    fn handle_research(&mut self, action: research::Action) {
        match action {
            research::Action::Rpc { method, params } => {
                // 改变项目状态的调用直接取回 State；其余结果交给研究面板。
                if matches!(method, "metadata.set" | "regions.add" | "regions.remove") { self.call(method, params); }
                else { self.call_value(method, params); }
            }
            research::Action::Job { method, params, label } => self.start_aux_job(method, params, label),
            research::Action::Locate(frame) => self.locate_frame(frame),
            research::Action::Compare { group_by, include_regions } => {
                let Some(paths) = rfd::FileDialog::new().add_filter("CadensView 项目", &["shotlab"]).pick_files() else { return; };
                if paths.len() < 2 { self.status = "至少选择两个项目".into(); return; }
                let paths: Vec<String> = paths.iter().map(|p| p.to_string_lossy().into_owned()).collect();
                self.status = format!("比较 {} 个项目…", paths.len());
                self.call_value("compare.run", json!({"paths": paths, "group_by": group_by, "exclude_regions": !include_regions}));
            }
            research::Action::ExportResearch => {
                let Some(path) = rfd::FileDialog::new().add_filter("CSV 表格", &["csv"])
                    .set_file_name("研究长表.csv").save_file() else { return; };
                match self.scope_params() {
                    Ok(mut params) => {
                        params["path"] = json!(path.to_string_lossy());
                        if self.motion_data.is_none() {
                            self.status = "提示：未运行“运动 · 拍数”，长表的拍数列为空".into();
                        }
                        self.call_quiet_result("research.export", params);
                    }
                    Err(error) => self.status = error,
                }
            }
            research::Action::RunVlm { force } => {
                let Ok(mut params) = self.scope_params() else {
                    self.status = "先标记镜头或切换到全片范围".into(); return;
                };
                let count = self.analysis_scope_ids().map(|ids| ids.len()).unwrap_or(self.state.completed_count as usize);
                let target = self.research.vlm_config.as_ref()
                    .map(|c| format!("{} · {}", c["provider"].as_str().unwrap_or(""), c["model"].as_str().unwrap_or("")))
                    .unwrap_or_else(|| "已配置的服务商".into());
                let confirmed = rfd::MessageDialog::new()
                    .set_title("发送关键帧到视觉大模型")
                    .set_description(format!("将把最多 {count} 个镜头、每个镜头约 3 张关键帧发送到 {target}，可能产生费用。\
已标注且边界未变的镜头会跳过{}。继续？", if force { "（本次强制重新标注）" } else { "" }))
                    .set_buttons(rfd::MessageButtons::YesNo).show() == rfd::MessageDialogResult::Yes;
                if !confirmed { return; }
                params["force"] = json!(force);
                self.start_aux_job("vlm.run", params, "视觉大模型标注");
            }
        }
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
            Ok(mut app) => {
                // 命令行/文件关联：cadensview.exe <项目.shotlab | 视频>
                if let Some(path) = std::env::args_os().nth(1).map(PathBuf::from).filter(|p| p.is_file()) {
                    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("shotlab")) { app.open_project(path); }
                    else { app.open_video(path); }
                }
                Ok(Box::new(app) as Box<dyn eframe::App>)
            }
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
