//! 音频播放：ffmpeg 解 PCM（f32le）→ cpal 输出。
//!
//! 设备与 cpal::Stream 在专属线程里持有（Stream 不保证 Send），控制全部走
//! 共享原子量 + 互斥队列。解码进程独立于视频流，起播用同一 `-ss` 对齐。

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// 缓冲上限（秒）：控制音画延迟。
const MAX_BUFFER_SECS: f32 = 0.35;

pub struct AudioHandle {
    buffer: Arc<Mutex<VecDeque<f32>>>,
    volume: Arc<AtomicU32>, // f32 bits
    muted: Arc<AtomicBool>,
    decoder: Arc<Mutex<Option<Child>>>,
    looping: Arc<AtomicBool>,
    generation: Arc<AtomicU32>,
    alive: Arc<AtomicBool>,
    sample_rate: u32,
    channels: u16,
}

impl AudioHandle {
    /// 打开默认输出设备；无设备时返回 None（GUI 静音运行，不报错）。
    pub fn spawn() -> Option<AudioHandle> {
        let (tx, rx) = std::sync::mpsc::channel::<Result<(u32, u16), String>>();
        let buffer = Arc::new(Mutex::new(VecDeque::<f32>::new()));
        let volume = Arc::new(AtomicU32::new(0.8f32.to_bits()));
        let muted = Arc::new(AtomicBool::new(false));
        let alive = Arc::new(AtomicBool::new(true));

        let (b, v, m, a) = (
            Arc::clone(&buffer),
            Arc::clone(&volume),
            Arc::clone(&muted),
            Arc::clone(&alive),
        );
        thread::spawn(move || {
            let host = cpal::default_host();
            let Some(device) = host.default_output_device() else {
                let _ = tx.send(Err("无音频输出设备".into()));
                return;
            };
            let supported = match device.default_output_config() {
                Ok(config) => config,
                Err(error) => {
                    let _ = tx.send(Err(error.to_string()));
                    return;
                }
            };
            let sample_rate = supported.sample_rate();
            let channels = supported.channels();
            let config: cpal::StreamConfig = supported.config();
            let format = supported.sample_format();
            let err_fn = |error| eprintln!("音频设备错误: {error}");

            let stream = match format {
                cpal::SampleFormat::F32 => {
                    let (b, v, m) = (Arc::clone(&b), Arc::clone(&v), Arc::clone(&m));
                    device.build_output_stream(
                        config.clone(),
                        move |data: &mut [f32], _| {
                            let mut queue = b.lock().unwrap();
                            let gain = if m.load(Ordering::Relaxed) {
                                0.0
                            } else {
                                f32::from_bits(v.load(Ordering::Relaxed))
                            };
                            for sample in data.iter_mut() {
                                *sample = queue.pop_front().unwrap_or(0.0) * gain;
                            }
                        },
                        err_fn,
                        None,
                    )
                }
                cpal::SampleFormat::I16 => {
                    let (b, v, m) = (Arc::clone(&b), Arc::clone(&v), Arc::clone(&m));
                    device.build_output_stream(
                        config.clone(),
                        move |data: &mut [i16], _| {
                            let mut queue = b.lock().unwrap();
                            let gain = if m.load(Ordering::Relaxed) {
                                0.0
                            } else {
                                f32::from_bits(v.load(Ordering::Relaxed))
                            };
                            for sample in data.iter_mut() {
                                let value = queue.pop_front().unwrap_or(0.0) * gain;
                                *sample = (value.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                            }
                        },
                        err_fn,
                        None,
                    )
                }
                other => {
                    let _ = tx.send(Err(format!("不支持的采样格式: {other:?}")));
                    return;
                }
            };
            let stream = match stream {
                Ok(stream) => stream,
                Err(error) => {
                    let _ = tx.send(Err(error.to_string()));
                    return;
                }
            };
            if let Err(error) = stream.play() {
                let _ = tx.send(Err(error.to_string()));
                return;
            }
            let _ = tx.send(Ok((sample_rate, channels)));
            // 持有 stream 直到 GUI 退出（Stream 不能跨线程移动）。
            while a.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(250));
            }
            drop(stream);
        });

        match rx.recv_timeout(Duration::from_secs(3)) {
            Ok(Ok((sample_rate, channels))) => Some(AudioHandle {
                buffer,
                volume,
                muted,
                decoder: Arc::new(Mutex::new(None)),
                looping: Arc::new(AtomicBool::new(false)),
                generation: Arc::new(AtomicU32::new(0)),
                alive,
                sample_rate,
                channels,
            }),
            Ok(Err(message)) => {
                eprintln!("音频不可用（静音运行）: {message}");
                None
            }
            Err(_) => {
                eprintln!("音频设备初始化超时（静音运行）");
                None
            }
        }
    }

    /// 从 start_s 起播；loop_duration 为 Some 时按段循环（原版 AVLoop 语义）。
    pub fn play(&self, video: &Path, start_s: f64) {
        self.start_decoder(video, start_s, None, false);
    }

    pub fn play_loop(&self, video: &Path, start_s: f64, duration_s: f64) {
        self.start_decoder(video, start_s, Some(duration_s), true);
    }

    pub fn stop(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.looping.store(false, Ordering::Relaxed);
        if let Some(mut child) = self.decoder.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.buffer.lock().unwrap().clear();
    }

    pub fn set_volume(&self, volume: f32) {
        self.volume
            .store(volume.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }

    pub fn set_muted(&self, muted: bool) {
        self.muted.store(muted, Ordering::Relaxed);
    }

    fn start_decoder(&self, video: &Path, start_s: f64, duration: Option<f64>, looping: bool) {
        self.stop();
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        self.looping.store(looping, Ordering::Relaxed);

        let video = video.to_path_buf();
        let buffer = Arc::clone(&self.buffer);
        let decoder = Arc::clone(&self.decoder);
        let looping_flag = Arc::clone(&self.looping);
        let generation_flag = Arc::clone(&self.generation);
        let alive = Arc::clone(&self.alive);
        let rate = self.sample_rate;
        let channels = self.channels;

        thread::spawn(move || {
            loop {
                if generation_flag.load(Ordering::SeqCst) != generation || !alive.load(Ordering::Relaxed) {
                    break;
                }
                let child = match spawn_decoder(&video, start_s, duration, rate, channels) {
                    Ok(child) => child,
                    Err(error) => {
                        eprintln!("音频解码启动失败: {error}");
                        break;
                    }
                };
                let mut child = child;
                let mut stdout = match child.stdout.take() {
                    Some(stdout) => stdout,
                    None => break,
                };
                *decoder.lock().unwrap() = Some(child);
                let capacity = (rate as f32 * channels as f32 * MAX_BUFFER_SECS) as usize;
                let mut raw = vec![0u8; 8192];
                let mut reached_eof = false;
                loop {
                    use std::io::Read;
                    match stdout.read(&mut raw) {
                        Ok(0) => {
                            reached_eof = true;
                            break;
                        }
                        Ok(count) => {
                            let mut queue = buffer.lock().unwrap();
                            for chunk in raw[..count].chunks_exact(4) {
                                if queue.len() >= capacity {
                                    break;
                                }
                                queue.push_back(f32::from_le_bytes([
                                    chunk[0], chunk[1], chunk[2], chunk[3],
                                ]));
                            }
                            drop(queue);
                            // 缓冲满时让出，避免把整段 PCM 一次性读进内存
                            if buffer.lock().unwrap().len() >= capacity {
                                thread::sleep(Duration::from_millis(20));
                            }
                        }
                        Err(_) => break,
                    }
                }
                if let Some(mut child) = decoder.lock().unwrap().take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                let still_current = generation_flag.load(Ordering::SeqCst) == generation;
                if !(reached_eof && looping_flag.load(Ordering::Relaxed) && still_current) {
                    break;
                }
                buffer.lock().unwrap().clear();
            }
        });
    }
}

impl Drop for AudioHandle {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Relaxed);
        self.stop();
    }
}

fn spawn_decoder(
    video: &PathBuf,
    start_s: f64,
    duration: Option<f64>,
    rate: u32,
    channels: u16,
) -> Result<Child, String> {
    let mut command = Command::new(ffmpeg_path());
    command
        .arg("-nostdin")
        .arg("-loglevel")
        .arg("error")
        .arg("-ss")
        .arg(format!("{start_s:.6}"));
    if let Some(duration) = duration {
        command.arg("-t").arg(format!("{duration:.6}"));
    }
    command
        .arg("-i")
        .arg(video)
        .args(["-map", "0:a:0", "-vn", "-f", "f32le"])
        .arg("-ac")
        .arg(channels.to_string())
        .arg("-ar")
        .arg(rate.to_string())
        .arg("-")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    command
        .spawn()
        .map_err(|error| format!("ffmpeg 启动失败: {error}"))
}

fn ffmpeg_path() -> PathBuf {
    if let Ok(path) = std::env::var("CADENSVIEW_FFMPEG") {
        return PathBuf::from(path);
    }
    if let Ok(exe) = std::env::current_exe() {
        for dir in exe.ancestors().skip(1) {
            let candidate = dir.join("tools").join("ffmpeg.exe");
            if candidate.exists() {
                return candidate;
            }
        }
    }
    PathBuf::from("ffmpeg")
}

/// 测试/诊断用：一次性解出全部 PCM（不经过音频设备）。
#[cfg(test)]
pub fn decode_probe(
    video: &Path,
    start_s: f64,
    duration: Option<f64>,
    rate: u32,
    channels: u16,
) -> Result<Vec<u8>, String> {
    let mut command = Command::new(ffmpeg_path());
    command
        .arg("-nostdin")
        .arg("-loglevel")
        .arg("error")
        .arg("-ss")
        .arg(format!("{start_s:.6}"));
    if let Some(duration) = duration {
        command.arg("-t").arg(format!("{duration:.6}"));
    }
    command
        .arg("-i")
        .arg(video)
        .args(["-map", "0:a:0", "-vn", "-f", "f32le"])
        .arg("-ac")
        .arg(channels.to_string())
        .arg("-ar")
        .arg(rate.to_string())
        .arg("-")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = command
        .output()
        .map_err(|error| format!("ffmpeg 启动失败: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg 解码失败: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(output.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_probe_reads_pcm_from_fixture() {
        let (Ok(video), Ok(ffmpeg)) = (
            std::env::var("CV_TEST_VIDEO"),
            std::env::var("CV_TEST_FFMPEG"),
        ) else {
            eprintln!("skipped: set CV_TEST_VIDEO/CV_TEST_FFMPEG to run");
            return;
        };
        std::env::set_var("CADENSVIEW_FFMPEG", ffmpeg);
        let pcm = decode_probe(Path::new(&video), 0.0, None, 48000, 2).expect("decode");
        assert!(pcm.len() > 4096, "PCM 数据过少: {} 字节", pcm.len());
        // f32 样本，应当有非零信号（夹具是 440Hz 正弦）
        let samples: Vec<f32> = pcm
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!(
            samples.iter().any(|s| s.abs() > 0.01),
            "PCM 全零，解码可能失败"
        );
    }
}
