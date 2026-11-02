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
    Show(u64), // 暂停态精确显示某帧
    Stop,
    Shutdown,
}

struct Stream {
    child: Child,
    next_frame: u64,   // 下一个要读的帧号
    started: Instant,  // 起播墙钟
    start_pts: i64,    // 起播帧 pts，用于节流对齐
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

impl PlayerHandle {
    pub fn spawn(ffmpeg: PathBuf, video: PathBuf, index: IndexInfo) -> PlayerHandle {
        let (tx, rx) = std::sync::mpsc::channel::<PlayerMsg>();
        let inner = Arc::new(Inner {
            cmd: Mutex::new(None),
            playing: Arc::new(AtomicBool::new(false)),
            pub_frame: Arc::new(AtomicU64::new(0)),
        });
        let worker = Arc::clone(&inner);
        thread::spawn(move || worker_loop(worker, ffmpeg, video, index, tx));
        PlayerHandle { inner, msgs: rx }
    }

    pub fn play(&self, frame: u64) {
        *self.inner.cmd.lock().unwrap() = Some(PlayerCmd::Play(frame));
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

    fn has_pending(&self) -> bool {
        self.inner.cmd.lock().unwrap().is_some()
    }
}

impl Drop for PlayerHandle {
    fn drop(&mut self) {
        *self.inner.cmd.lock().unwrap() = Some(PlayerCmd::Shutdown);
    }
}

fn worker_loop(inner: Arc<Inner>, ffmpeg: PathBuf, video: PathBuf, index: IndexInfo, tx: std::sync::mpsc::Sender<PlayerMsg>) {
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
                        let _ = tx.send(PlayerMsg::Frame { frame, width, height, rgba });
                    }
                    Err(error) => {
                        let _ = tx.send(PlayerMsg::Failed(error));
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
                            });
                        }
                        Err(error) => {
                            let _ = tx.send(PlayerMsg::Failed(error.to_string()));
                        }
                    }
                }
                PlayerCmd::Shutdown => unreachable!(),
            }
            continue;
        }

        match stream.as_mut() {
            Some(state) => {
                if !pump_frame(state, &index, &inner, &tx) {
                    stream = None;
                    inner.playing.store(false, Ordering::SeqCst);
                    let _ = tx.send(PlayerMsg::Ended);
                }
            }
            None => thread::sleep(Duration::from_millis(10)),
        }
    }
}

/// 读一帧、按 pts 节流后发出。false = 流结束或出错。
fn pump_frame(state: &mut Stream, index: &IndexInfo, inner: &Inner, tx: &std::sync::mpsc::Sender<PlayerMsg>) -> bool {
    let (width, height) = index.size();
    if width == 0 || height == 0 {
        let _ = tx.send(PlayerMsg::Failed("索引缺少画面尺寸".into()));
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
    tx.send(PlayerMsg::Frame { frame, width, height, rgba }).is_ok()
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
