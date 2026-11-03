//! FFmpeg 播放器：连续播放 = rawvideo 流（按 pts 节流），逐帧/seek = 单次 -ss 精确取帧。
//! 解码跑在专属线程；UI 只收 Frame 消息并上传纹理。

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::model::IndexInfo;

pub enum PlayerMsg {
    Frame { frame: u64, width: usize, height: usize, rgba: Vec<u8> },
    Ended,
    Failed(String),
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PlayerCmd {
    Play(u64), // 从某帧连续播放（静音分析模式）
    PlayLoop(u64, u64), // 局部循环：在 [start, end) 帧范围内反复
    Show(u64), // 暂停态精确显示某帧
    Stop,
    Shutdown,
}

struct Stream {
    child: Arc<Mutex<Child>>,
    stdout: ChildStdout,
    generation: u64,
    next_frame: u64, // -ss from 的首个输出就是 from
    started: Option<Instant>, // 首帧解出后起算，不把 ffmpeg 启动耗时算入播放
    start_pts: i64,
    loop_end: Option<u64>,
    loop_start: u64,
}

impl Drop for Stream {
    fn drop(&mut self) {
        terminate(&self.child);
    }
}

struct Control {
    cmd: Option<PlayerCmd>,
    generation: u64,
    child: Option<Arc<Mutex<Child>>>,
}

struct Inner {
    control: Mutex<Control>,
    playing: AtomicBool,
    pub_frame: AtomicU64,
}

impl Inner {
    fn is_current(&self, generation: u64) -> bool {
        self.control.lock().unwrap().generation == generation
    }
}

pub struct PlayerHandle {
    inner: Arc<Inner>,
    pub msgs: std::sync::mpsc::Receiver<PlayerMsg>,
    worker: Option<thread::JoinHandle<()>>,
}

/// 帧/状态送达后唤醒 UI 线程（egui 空闲时不会自轮询，必须显式请求重绘）。
pub type Waker = Arc<dyn Fn() + Send + Sync>;

impl PlayerHandle {
    pub fn spawn(ffmpeg: PathBuf, video: PathBuf, index: IndexInfo, wake: Waker) -> PlayerHandle {
        let (tx, rx) = std::sync::mpsc::channel::<PlayerMsg>();
        let inner = Arc::new(Inner {
            control: Mutex::new(Control { cmd: None, generation: 0, child: None }),
            playing: AtomicBool::new(false),
            pub_frame: AtomicU64::new(0),
        });
        let worker_inner = Arc::clone(&inner);
        let worker = thread::spawn(move || worker_loop(worker_inner, ffmpeg, video, index, Out { tx, wake }));
        PlayerHandle { inner, msgs: rx, worker: Some(worker) }
    }

    fn request(&self, cmd: PlayerCmd) {
        let child = {
            let mut control = self.inner.control.lock().unwrap();
            control.generation = control.generation.wrapping_add(1);
            control.cmd = Some(cmd);
            self.inner.playing.store(false, Ordering::SeqCst);
            control.child.take()
        };
        // 不能只投命令：worker 可能正阻塞在 stdout.read_exact。
        if let Some(child) = child {
            terminate(&child);
        }
    }

    pub fn play(&self, frame: u64) {
        self.request(PlayerCmd::Play(frame));
    }

    /// 局部循环播放：到达 end 后自动从 start 重启（原版 transport.play repeat 语义）。
    pub fn play_loop(&self, start: u64, end: u64) {
        self.request(PlayerCmd::PlayLoop(start, end));
    }

    pub fn show(&self, frame: u64) {
        self.request(PlayerCmd::Show(frame));
    }

    pub fn stop(&self) {
        self.request(PlayerCmd::Stop);
    }

    pub fn is_playing(&self) -> bool {
        self.inner.playing.load(Ordering::SeqCst)
    }

    pub fn frame(&self) -> u64 {
        self.inner.pub_frame.load(Ordering::SeqCst)
    }
}

impl Drop for PlayerHandle {
    fn drop(&mut self) {
        self.request(PlayerCmd::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn worker_loop(inner: Arc<Inner>, ffmpeg: PathBuf, video: PathBuf, index: IndexInfo, out: Out) {
    let mut stream: Option<Stream> = None;
    loop {
        let command = {
            let mut control = inner.control.lock().unwrap();
            control.cmd.take().map(|cmd| (cmd, control.generation))
        };
        if let Some((cmd, generation)) = command {
            clear_stream(&inner, &mut stream);
            if cmd == PlayerCmd::Shutdown {
                return;
            }
            if cmd == PlayerCmd::Stop {
                continue;
            }
            let (from, loop_end, playing) = match cmd {
                PlayerCmd::Show(frame) => (frame, None, false),
                PlayerCmd::Play(frame) => (frame, None, true),
                PlayerCmd::PlayLoop(start, end) => (start, Some(end), true),
                _ => unreachable!(),
            };
            match start_stream(&ffmpeg, &video, &index, from, loop_end, generation) {
                Ok(mut state) => {
                    if !register_stream(&inner, &state, playing) {
                        continue; // seek/stop 与 spawn 并发；Drop 回收这个过期 child
                    }
                    if playing {
                        stream = Some(state);
                    } else {
                        match pump_frame(&mut state, &index, &inner, &out) {
                            Pump::Failed(error) => { out.send_current(&inner, generation, PlayerMsg::Failed(error)); }
                            Pump::Disconnected => return,
                            _ => {}
                        }
                        clear_stream(&inner, &mut Some(state));
                    }
                }
                Err(error) => finish(&inner, &out, generation, Some(error), playing),
            }
            continue;
        }

        let Some(state) = stream.as_mut() else {
            thread::sleep(Duration::from_millis(10));
            continue;
        };
        let generation = state.generation;
        let loop_end = state.loop_end;
        let restart = state.loop_start;
        match pump_frame(state, &index, &inner, &out) {
            Pump::Frame => {}
            Pump::Interrupted => clear_stream(&inner, &mut stream),
            Pump::Disconnected => {
                clear_stream(&inner, &mut stream);
                return;
            }
            Pump::Failed(error) => {
                clear_stream(&inner, &mut stream);
                // 解码失败不能当作正常 EOF 无限重启。
                finish(&inner, &out, generation, Some(error), true);
            }
            Pump::Finished => {
                clear_stream(&inner, &mut stream);
                if !inner.is_current(generation) {
                    continue;
                }
                if let Some(end) = loop_end {
                    match start_stream(&ffmpeg, &video, &index, restart, Some(end), generation) {
                        Ok(state) => {
                            if register_stream(&inner, &state, true) {
                                stream = Some(state);
                            }
                        }
                        Err(error) => finish(&inner, &out, generation, Some(error), true),
                    }
                } else {
                    finish(&inner, &out, generation, None, true);
                }
            }
        }
    }
}

fn register_stream(inner: &Inner, state: &Stream, playing: bool) -> bool {
    let mut control = inner.control.lock().unwrap();
    if control.generation != state.generation {
        return false;
    }
    control.child = Some(Arc::clone(&state.child));
    inner.playing.store(playing, Ordering::SeqCst);
    true
}

fn clear_stream(inner: &Inner, stream: &mut Option<Stream>) {
    if let Some(state) = stream.take() {
        let mut control = inner.control.lock().unwrap();
        if control.child.as_ref().is_some_and(|child| Arc::ptr_eq(child, &state.child)) {
            control.child = None;
        }
        drop(control);
        drop(state);
    }
}

fn finish(inner: &Inner, out: &Out, generation: u64, error: Option<String>, ended: bool) {
    {
        let control = inner.control.lock().unwrap();
        if control.generation != generation {
            return;
        }
        inner.playing.store(false, Ordering::SeqCst);
    }
    if let Some(error) = error {
        out.send_current(inner, generation, PlayerMsg::Failed(error));
    }
    if ended {
        out.send_current(inner, generation, PlayerMsg::Ended);
    }
}

/// 发消息并唤醒 UI：egui 空闲时不轮询，缺了唤醒帧会"停在旧画面"。
struct Out {
    tx: std::sync::mpsc::Sender<PlayerMsg>,
    wake: Waker,
}

impl Out {
    fn send_current(&self, inner: &Inner, generation: u64, message: PlayerMsg) -> bool {
        let ok = {
            let control = inner.control.lock().unwrap();
            if control.generation != generation {
                return false;
            }
            if let PlayerMsg::Frame { frame, .. } = &message {
                inner.pub_frame.store(*frame, Ordering::SeqCst);
            }
            self.tx.send(message).is_ok()
        };
        (self.wake)();
        ok
    }
}

#[derive(Debug, PartialEq)]
enum Pump {
    Frame,
    Finished,
    Interrupted,
    Disconnected,
    Failed(String),
}

fn start_stream(ffmpeg: &PathBuf, video: &PathBuf, index: &IndexInfo, from: u64,
    loop_end: Option<u64>, generation: u64) -> Result<Stream, String>
{
    validate_index(index)?;
    let from = from.min(index.frames() - 1);
    let loop_end = loop_end.map(|end| end.min(index.frames()).max(from + 1));
    let mut child = spawn_stream(ffmpeg, video, index, from)
        .map_err(|error| format!("ffmpeg 启动失败: {error}"))?;
    let Some(stdout) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err("ffmpeg 未打开视频输出管道".into());
    };
    Ok(Stream {
        child: Arc::new(Mutex::new(child)), stdout, generation, next_frame: from,
        started: None, start_pts: index.pts[from as usize], loop_end, loop_start: from,
    })
}

/// 边界使用下帧 PTS；片尾使用最后一帧时长，保证 [start, end) 的末帧也完整播放。
fn target_seconds(index: &IndexInfo, frame: u64, start_pts: i64) -> f64 {
    let (num, den) = index.time_base();
    let tick_s = num as f64 / den as f64;
    if let Some(pts) = index.pts.get(frame as usize) {
        return (*pts as i128 - start_pts as i128) as f64 * tick_s;
    }
    let last = index.pts.len() - 1;
    let duration = index.durations.get(last).copied().filter(|duration| *duration > 0)
        .or_else(|| last.checked_sub(1).map(|prev| index.pts[last].saturating_sub(index.pts[prev])).filter(|d| *d > 0));
    let tail_s = duration.map(|duration| duration as f64 * tick_s).unwrap_or_else(|| {
        let mut rate = index.metadata.average_rate.as_deref().unwrap_or("24").split('/');
        let num = rate.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(24.0);
        let den = rate.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.0);
        let seconds = den / num;
        if seconds.is_finite() && seconds > 0.0 { seconds } else { 1.0 / 24.0 }
    });
    (index.pts[last] as i128 - start_pts as i128) as f64 * tick_s + tail_s
}

fn wait_for_frame(state: &Stream, index: &IndexInfo, inner: &Inner) -> bool {
    let Some(started) = state.started else { return inner.is_current(state.generation) };
    let target = target_seconds(index, state.next_frame, state.start_pts).max(0.0);
    loop {
        if !inner.is_current(state.generation) {
            return false;
        }
        let remaining = target - started.elapsed().as_secs_f64();
        if remaining <= 0.0 {
            return true;
        }
        thread::sleep(Duration::from_secs_f64(remaining.min(0.01)));
    }
}

/// 读取 rawvideo 后按真实帧号发消息；索引/循环边界在读下一帧之前检查。
fn pump_frame(state: &mut Stream, index: &IndexInfo, inner: &Inner, out: &Out) -> Pump {
    if !inner.is_current(state.generation) {
        return Pump::Interrupted;
    }
    if state.next_frame >= state.loop_end.unwrap_or(index.frames()) {
        return if wait_for_frame(state, index, inner) { Pump::Finished } else { Pump::Interrupted };
    }
    let (width, height) = index.size();
    let expected = match image_bytes(width, height, 4) {
        Ok(expected) => expected,
        Err(error) => return Pump::Failed(error),
    };
    let mut rgba = vec![0u8; expected];
    if let Err(error) = state.stdout.read_exact(&mut rgba) {
        return if inner.is_current(state.generation) {
            Pump::Failed(format!("ffmpeg 在帧 {} 提前结束或读取失败: {error}", state.next_frame))
        } else { Pump::Interrupted };
    }
    state.started.get_or_insert_with(Instant::now);
    if !wait_for_frame(state, index, inner) {
        return Pump::Interrupted;
    }
    let frame = state.next_frame;
    state.next_frame += 1;
    if out.send_current(inner, state.generation, PlayerMsg::Frame { frame, width, height, rgba }) {
        Pump::Frame
    } else if inner.is_current(state.generation) {
        Pump::Disconnected
    } else { Pump::Interrupted }
}

fn image_bytes(width: usize, height: usize, channels: usize) -> Result<usize, String> {
    if width == 0 || height == 0 {
        return Err("画面尺寸必须大于零".into());
    }
    width.checked_mul(height).and_then(|n| n.checked_mul(channels))
        .filter(|n| *n <= isize::MAX as usize).ok_or_else(|| "画面尺寸过大".into())
}

fn validate_index(index: &IndexInfo) -> Result<(), String> {
    let (width, height) = index.size();
    image_bytes(width, height, 4)?;
    if index.frames() == 0 {
        return Err("索引没有视频帧".into());
    }
    let (num, den) = index.time_base();
    if num <= 0 || den <= 0 {
        return Err("索引 time_base 无效".into());
    }
    Ok(())
}

fn spawn_stream(ffmpeg: &PathBuf, video: &PathBuf, index: &IndexInfo, from: u64) -> std::io::Result<Child> {
    let seconds = index.seconds(from);
    Command::new(ffmpeg)
        .arg("-nostdin")
        .arg("-loglevel").arg("error")
        .arg("-ss").arg(format!("{seconds:.6}"))
        .arg("-i").arg(video)
        .args(["-map", "0:v:0", "-an", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
}

/// 线程友好的单帧采集（自带 index 拷贝，给 A/B 对照等离屏使用）。
pub fn capture_frame(ffmpeg: &PathBuf, video: &PathBuf, index: IndexInfo, frame: u64) -> Result<(usize, usize, Vec<u8>), String> {
    capture_one(ffmpeg, video, &index, frame)
}

fn capture_one(ffmpeg: &PathBuf, video: &PathBuf, index: &IndexInfo, frame: u64) -> Result<(usize, usize, Vec<u8>), String> {
    validate_index(index)?;
    let seconds = index.seconds(frame.min(index.frames() - 1));
    let (width, height) = index.size();
    let output = Command::new(ffmpeg)
        .arg("-nostdin")
        .arg("-loglevel").arg("error")
        .arg("-ss").arg(format!("{seconds:.6}"))
        .arg("-i").arg(video)
        .args(["-map", "0:v:0", "-an", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("ffmpeg 启动失败: {e}"))?;
    if !output.status.success() {
        return Err(format!("ffmpeg 取帧失败: {}", String::from_utf8_lossy(&output.stderr).trim()));
    }
    let expected = image_bytes(width, height, 4)?;
    if output.stdout.len() != expected {
        return Err(format!("ffmpeg 帧长度不符 ({} != {})", output.stdout.len(), expected));
    }
    Ok((width, height, output.stdout[..expected].to_vec()))
}

/// 小尺寸单帧采集（rgb24）：缩略图与景别预处理复用。
pub fn capture_small(ffmpeg: &PathBuf, video: &PathBuf, seconds: f64, width: u32, height: u32) -> Result<Vec<u8>, String> {
    let expected = image_bytes(width as usize, height as usize, 3)?;
    if !seconds.is_finite() || seconds < 0.0 {
        return Err("缩略图时间必须为非负有限值".into());
    }
    let scale = format!("scale={width}:{height}");
    let output = Command::new(ffmpeg)
        .arg("-nostdin")
        .arg("-loglevel").arg("error")
        .arg("-ss").arg(format!("{seconds:.6}"))
        .arg("-i").arg(video)
        .args(["-map", "0:v:0", "-an", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24"])
        .arg("-vf").arg(&scale)
        .arg("-")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("ffmpeg 启动失败: {e}"))?;
    if !output.status.success() {
        return Err(format!("ffmpeg 缩略图失败: {}", String::from_utf8_lossy(&output.stderr).trim()));
    }
    if output.stdout.len() != expected {
        return Err(format!("缩略图帧长度不符 ({} != {})", output.stdout.len(), expected));
    }
    Ok(output.stdout[..expected].to_vec())
}

/// rgb24 → rgba（简单通道扩展，小图无所谓性能）。
pub fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for pixel in rgb.chunks_exact(3) {
        rgba.extend_from_slice(pixel);
        rgba.push(255);
    }
    rgba
}

fn terminate(child: &Arc<Mutex<Child>>) {
    let mut child = child.lock().unwrap();
    let _ = child.kill();
    let _ = child.wait();
}

/// 播放时让检测 worker 让路的标记文件（协议见 shotlab/onnx_worker.py）。
pub struct PlaybackMarker {
    path: std::path::PathBuf,
    active: bool,
}

impl PlaybackMarker {
    pub fn new(index_dir: &std::path::Path) -> PlaybackMarker {
        PlaybackMarker { path: index_dir.join("playback"), active: false }
    }

    pub fn set(&mut self, playing: bool) {
        if playing == self.active {
            return;
        }
        if playing {
            let _ = std::fs::File::create(&self.path);
        } else {
            let _ = std::fs::remove_file(&self.path);
        }
        self.active = playing;
    }
}

impl Drop for PlaybackMarker {
    fn drop(&mut self) {
        self.set(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 需要合成视频夹具；未设置环境变量时跳过（见 tests/make_fixture.py）。
    #[test]
    fn capture_is_frame_accurate_and_stream_aligns() {
        let (Ok(video), Ok(index_json), Ok(ffmpeg)) = (
            std::env::var("CV_TEST_VIDEO"),
            std::env::var("CV_TEST_INDEX"),
            std::env::var("CV_TEST_FFMPEG"),
        ) else {
            eprintln!("skipped: set CV_TEST_VIDEO/CV_TEST_INDEX/CV_TEST_FFMPEG to run");
            return;
        };
        let index: crate::model::IndexInfo =
            serde_json::from_str(&std::fs::read_to_string(index_json).unwrap()).unwrap();
        let video = PathBuf::from(video);
        let ffmpeg = PathBuf::from(ffmpeg);

        // 合成夹具：帧 0-23 红、24-47 蓝。色彩经 yuv 往返有偏差（夹具元数据不自洽），
        // 断言按色系而非精确 RGB；23/24 是帧号精度的最紧边界。
        fn family(rgba: &[u8], w: usize, h: usize) -> [u8; 3] {
            let center = ((h / 2) * w + w / 2) * 4;
            [rgba[center], rgba[center + 1], rgba[center + 2]]
        }
        fn assert_red(got: [u8; 3], context: &str) {
            assert!(got[0] > 200 && got[2] < 80, "{context}: 应为红色系，实得 {got:?}");
        }
        fn assert_blue(got: [u8; 3], context: &str) {
            assert!(got[2] > 200 && got[0] < 80, "{context}: 应为蓝色系，实得 {got:?}");
        }

        for (frame, red) in [(0u64, true), (23, true), (24, false), (30, false), (47, false)] {
            let (w, h, rgba) = capture_one(&ffmpeg, &video, &index, frame).expect("capture");
            let got = family(&rgba, w, h);
            if red { assert_red(got, &format!("单帧 {frame}")); } else { assert_blue(got, &format!("单帧 {frame}")); }
        }

        // 经真实 worker/pump_frame 验证消息中的帧号，不能只验证 stdout 像素。
        let player = PlayerHandle::spawn(ffmpeg, video, index, Arc::new(|| {}));
        player.play(22);
        for (step, red) in [(0u64, true), (1, true), (2, false), (3, false)] {
            match player.msgs.recv_timeout(Duration::from_secs(3)).expect("stream message") {
                PlayerMsg::Frame { frame, width, height, rgba } => {
                    assert_eq!(frame, 22 + step, "pump_frame 帧号偏移");
                    let got = family(&rgba, width, height);
                    if red { assert_red(got, &format!("流式帧 {frame}")); } else { assert_blue(got, &format!("流式帧 {frame}")); }
                }
                PlayerMsg::Failed(error) => panic!("stream failed: {error}"),
                PlayerMsg::Ended => panic!("流结束过早"),
            }
        }
    }

    #[test]
    fn capture_small_returns_rgb_from_fixture() {
        let (Ok(video), Ok(ffmpeg)) = (
            std::env::var("CV_TEST_VIDEO"),
            std::env::var("CV_TEST_FFMPEG"),
        ) else {
            eprintln!("skipped: set CV_TEST_VIDEO/CV_TEST_FFMPEG to run");
            return;
        };
        for (seconds, red) in [(0.0, true), (1.0, false)] {
            let rgb = capture_small(&PathBuf::from(&ffmpeg), &PathBuf::from(&video), seconds, 32, 18)
                .expect("capture_small must write RGB to stdout");
            assert_eq!(rgb.len(), 32 * 18 * 3);
            let center = ((18 / 2) * 32 + 32 / 2) * 3;
            let got = &rgb[center..center + 3];
            assert!(if red { got[0] > 200 && got[2] < 80 } else { got[2] > 200 && got[0] < 80 },
                "缩略图 {seconds}s 颜色错误: {got:?}");
        }
    }
}
