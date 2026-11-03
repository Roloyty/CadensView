//! FFmpeg 播放器：连续播放 = rawvideo 流（按 pts 节流），逐帧/seek = 单次 -ss 精确取帧。
//! 解码跑在专属线程；UI 只收 Frame 消息并上传纹理。

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
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
    child: Child,
    next_frame: u64,   // 下一个要读的帧号
    started: Instant,  // 起播墙钟
    start_pts: i64,    // 起播帧 pts，用于节流对齐
    loop_end: Option<u64>, // 循环终点（到达后从 loop_start 重新开始）
    loop_start: u64,   // 循环起点
}

struct Inner {
    cmd: Mutex<Option<PlayerCmd>>,
    playing: Arc<AtomicBool>,
    pub_frame: Arc<AtomicU64>,
}

pub struct PlayerHandle {
    inner: Arc<Inner>,
    pub msgs: std::sync::mpsc::Receiver<PlayerMsg>,
}

/// 帧/状态送达后唤醒 UI 线程（egui 空闲时不会自轮询，必须显式请求重绘）。
pub type Waker = Arc<dyn Fn() + Send + Sync>;

impl PlayerHandle {
    pub fn spawn(ffmpeg: PathBuf, video: PathBuf, index: IndexInfo, wake: Waker) -> PlayerHandle {
        let (tx, rx) = std::sync::mpsc::channel::<PlayerMsg>();
        let inner = Arc::new(Inner {
            cmd: Mutex::new(None),
            playing: Arc::new(AtomicBool::new(false)),
            pub_frame: Arc::new(AtomicU64::new(0)),
        });
        let worker = Arc::clone(&inner);
        thread::spawn(move || worker_loop(worker, ffmpeg, video, index, Out { tx, wake }));
        PlayerHandle { inner, msgs: rx }
    }

    pub fn play(&self, frame: u64) {
        *self.inner.cmd.lock().unwrap() = Some(PlayerCmd::Play(frame));
    }

    /// 局部循环播放：到达 end 后自动从 start 重启（原版 transport.play repeat 语义）。
    pub fn play_loop(&self, start: u64, end: u64) {
        *self.inner.cmd.lock().unwrap() = Some(PlayerCmd::PlayLoop(start, end));
    }

    pub fn show(&self, frame: u64) {
        *self.inner.cmd.lock().unwrap() = Some(PlayerCmd::Show(frame));
    }

    pub fn stop(&self) {
        *self.inner.cmd.lock().unwrap() = Some(PlayerCmd::Stop);
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
        *self.inner.cmd.lock().unwrap() = Some(PlayerCmd::Shutdown);
    }
}

fn worker_loop(inner: Arc<Inner>, ffmpeg: PathBuf, video: PathBuf, index: IndexInfo, out: Out) {
    let mut stream: Option<Stream> = None;
    loop {
        // 命令到达时立刻打断当前流；否则继续泵流或小睡。
        let cmd = if inner.cmd.lock().unwrap().is_some() {
            inner.cmd.lock().unwrap().take()
        } else {
            None
        };
        if let Some(cmd) = cmd {
            if matches!(cmd, PlayerCmd::Shutdown) {
                kill(&mut stream);
                return;
            }
            kill(&mut stream);
            inner.playing.store(false, Ordering::SeqCst);
            match cmd {
                PlayerCmd::Stop => {}
                PlayerCmd::Show(frame) => match capture_one(&ffmpeg, &video, &index, frame) {
                    Ok((width, height, rgba)) => {
                        let frame = frame.min(index.frames().saturating_sub(1));
                        inner.pub_frame.store(frame, Ordering::SeqCst);
                        out.send(PlayerMsg::Frame { frame, width, height, rgba });
                    }
                    Err(error) => {
                        out.send(PlayerMsg::Failed(error));
                    }
                },
                PlayerCmd::Play(from) => {
                    let from = from.min(index.frames().saturating_sub(1));
                    match spawn_stream(&ffmpeg, &video, &index, from) {
                        Ok(child) => {
                            inner.playing.store(true, Ordering::SeqCst);
                            inner.pub_frame.store(from, Ordering::SeqCst);
                            stream = Some(Stream {
                                child,
                                next_frame: from + 1,
                                started: Instant::now(),
                                start_pts: index.pts.get(from as usize).copied().unwrap_or(0),
                                loop_end: None,
                                loop_start: from,
                            });
                        }
                        Err(error) => {
                            out.send(PlayerMsg::Failed(error.to_string()));
                        }
                    }
                }
                PlayerCmd::PlayLoop(start, end) => {
                    let start = start.min(index.frames().saturating_sub(1));
                    let end = end.min(index.frames()).max(start + 1);
                    match spawn_stream(&ffmpeg, &video, &index, start) {
                        Ok(child) => {
                            inner.playing.store(true, Ordering::SeqCst);
                            inner.pub_frame.store(start, Ordering::SeqCst);
                            stream = Some(Stream {
                                child,
                                next_frame: start + 1,
                                started: Instant::now(),
                                start_pts: index.pts.get(start as usize).copied().unwrap_or(0),
                                loop_end: Some(end),
                                loop_start: start,
                            });
                        }
                        Err(error) => {
                            out.send(PlayerMsg::Failed(error.to_string()));
                        }
                    }
                }
                PlayerCmd::Shutdown => unreachable!(),
            }
            continue;
        }

        match stream.as_mut() {
            Some(state) => {
                // 循环模式：到达终点时重启 ffmpeg 流从起点
                if let Some(loop_end) = state.loop_end {
                    if state.next_frame >= loop_end {
                        let restart = state.loop_start;
                        kill(&mut stream);
                        inner.playing.store(false, Ordering::SeqCst);
                        match spawn_stream(&ffmpeg, &video, &index, restart) {
                            Ok(child) => {
                                inner.playing.store(true, Ordering::SeqCst);
                                inner.pub_frame.store(restart, Ordering::SeqCst);
                                stream = Some(Stream {
                                    child,
                                    next_frame: restart + 1,
                                    started: Instant::now(),
                                    start_pts: index.pts.get(restart as usize).copied().unwrap_or(0),
                                    loop_end: Some(loop_end),
                                    loop_start: restart,
                                });
                            }
                            Err(error) => {
                                out.send(PlayerMsg::Failed(error.to_string()));
                            }
                        }
                        continue;
                    }
                }
                if !pump_frame(state, &index, &inner, &out) {
                    // 流结束：循环模式下自动重启
                    if let Some(loop_end) = state.loop_end {
                        let restart = state.loop_start;
                        kill(&mut stream);
                        match spawn_stream(&ffmpeg, &video, &index, restart) {
                            Ok(child) => {
                                inner.playing.store(true, Ordering::SeqCst);
                                inner.pub_frame.store(restart, Ordering::SeqCst);
                                stream = Some(Stream {
                                    child,
                                    next_frame: restart + 1,
                                    started: Instant::now(),
                                    start_pts: index.pts.get(restart as usize).copied().unwrap_or(0),
                                    loop_end: Some(loop_end),
                                    loop_start: restart,
                                });
                                continue;
                            }
                            Err(_) => {}
                        }
                    }
                    stream = None;
                    inner.playing.store(false, Ordering::SeqCst);
                    out.send(PlayerMsg::Ended);
                }
            }
            None => thread::sleep(Duration::from_millis(10)),
        }
    }
}

/// 发消息并唤醒 UI：egui 空闲时不轮询，缺了唤醒帧会"停在旧画面"。
struct Out {
    tx: std::sync::mpsc::Sender<PlayerMsg>,
    wake: Waker,
}

impl Out {
    fn send(&self, message: PlayerMsg) -> bool {
        let ok = self.tx.send(message).is_ok();
        (self.wake)();
        ok
    }
}

/// 读一帧、按 pts 节流后发出。false = 流结束或出错。
fn pump_frame(state: &mut Stream, index: &IndexInfo, inner: &Inner, out: &Out) -> bool {
    let (width, height) = index.size();
    if width == 0 || height == 0 {
        out.send(PlayerMsg::Failed("索引缺少画面尺寸".into()));
        return false;
    }
    let mut rgba = vec![0u8; width * height * 4];
    let stdout = match state.child.stdout.as_mut() {
        Some(out) => out,
        None => return false,
    };
    if !read_exact_or_eof(stdout, &mut rgba) {
        return false;
    }
    // 按 pts 差对齐墙钟：解码快于播放节奏时等待。
    let (num, den) = index.time_base();
    let target = index
        .pts
        .get(state.next_frame as usize)
        .map(|pts| (*pts - state.start_pts) as f64 * num as f64 / den as f64)
        .unwrap_or_else(|| state.started.elapsed().as_secs_f64());
    let elapsed = state.started.elapsed().as_secs_f64();
    if target > elapsed {
        thread::sleep(Duration::from_secs_f64((target - elapsed).min(0.25)));
    }
    let frame = state.next_frame;
    state.next_frame += 1;
    inner.pub_frame.store(frame, Ordering::SeqCst);
    out.send(PlayerMsg::Frame { frame, width, height, rgba })
}

fn spawn_stream(ffmpeg: &PathBuf, video: &PathBuf, index: &IndexInfo, from: u64) -> std::io::Result<Child> {
    let seconds = index.seconds(from);
    Command::new(ffmpeg)
        .arg("-nostdin")
        .arg("-loglevel").arg("error")
        .arg("-ss").arg(format!("{seconds:.6}"))
        .arg("-i").arg(video)
        .args(["-map", "0:v:0", "-an", "-f", "rawvideo", "-pix_fmt", "rgba", "-"])
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
    let seconds = index.seconds(frame);
    let (width, height) = index.size();
    if width == 0 || height == 0 {
        return Err("索引缺少画面尺寸".into());
    }
    let output = Command::new(ffmpeg)
        .arg("-nostdin")
        .arg("-loglevel").arg("error")
        .arg("-ss").arg(format!("{seconds:.6}"))
        .arg("-i").arg(video)
        .args(["-map", "0:v:0", "-an", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("ffmpeg 启动失败: {e}"))?;
    let expected = width * height * 4;
    if output.stdout.len() < expected {
        return Err(format!("ffmpeg 未返回完整帧 ({} < {})", output.stdout.len(), expected));
    }
    Ok((width, height, output.stdout[..expected].to_vec()))
}

/// 小尺寸单帧采集（rgb24）：缩略图与景别预处理复用。
pub fn capture_small(ffmpeg: &PathBuf, video: &PathBuf, seconds: f64, width: u32, height: u32) -> Result<Vec<u8>, String> {
    let scale = format!("scale={width}:{height}");
    let output = Command::new(ffmpeg)
        .arg("-nostdin")
        .arg("-loglevel").arg("error")
        .arg("-ss").arg(format!("{seconds:.6}"))
        .arg("-i").arg(video)
        .args(["-map", "0:v:0", "-an", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24"])
        .arg("-vf").arg(&scale)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("ffmpeg 启动失败: {e}"))?;
    let expected = (width as usize) * (height as usize) * 3;
    if output.stdout.len() < expected {
        return Err(format!("缩略图帧不完整 ({} < {})", output.stdout.len(), expected));
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

fn read_exact_or_eof(stdout: &mut impl Read, buffer: &mut [u8]) -> bool {
    let mut filled = 0;
    while filled < buffer.len() {
        match stdout.read(&mut buffer[filled..]) {
            Ok(0) => return false,
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return false,
        }
    }
    true
}

fn kill(stream: &mut Option<Stream>) {
    if let Some(mut state) = stream.take() {
        let _ = state.child.kill();
        let _ = state.child.wait();
    }
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

        // 流式播放：从帧 22 起连续读 4 帧，应跨过 23/24 边界：红红蓝蓝
        let mut stream = spawn_stream(&ffmpeg, &video, &index, 22).expect("stream");
        let (w, h) = index.size();
        for (step, red) in [(0u64, true), (1, true), (2, false), (3, false)] {
            let mut rgba = vec![0u8; w * h * 4];
            assert!(read_exact_or_eof(stream.stdout.as_mut().unwrap(), &mut rgba), "流结束过早");
            let got = family(&rgba, w, h);
            let frame = 22 + step;
            if red { assert_red(got, &format!("流式帧 {frame}")); } else { assert_blue(got, &format!("流式帧 {frame}")); }
        }
        let _ = stream.kill();
    }
}
