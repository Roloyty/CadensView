//! 镜头缩略图：串行 worker + 磁盘缓存。
//! ffmpeg 每次启动约 50-150ms，一个一个来；命中缓存则零开销。
//! 缓存文件 = %TEMP%/cadensview/thumbs/<键>.bin，头部 8 字节 (w,u32 + h,u32) + RGBA。

use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::thread;

use crate::player;

pub struct ThumbRequest {
    /// 稳定键：视频指纹 + 镜头边界（边界移动后自动失效）。
    pub key: String,
    pub video: String,
    pub seconds: f64,
    pub width: u32,
    pub height: u32,
}

pub enum ThumbOutcome {
    Ready { key: String, width: u32, height: u32, rgba: Vec<u8> },
    Failed { key: String, message: String },
}

/// 启动串行缩略图 worker；返回任务投递端。
pub fn spawn_worker(ffmpeg: PathBuf, done: Sender<ThumbOutcome>, wake: player::Waker) -> Sender<ThumbRequest> {
    let (tx, rx) = std::sync::mpsc::channel::<ThumbRequest>();
    let cache_dir = std::env::temp_dir().join("cadensview").join("thumbs");
    let _ = std::fs::create_dir_all(&cache_dir);
    thread::spawn(move || {
        for request in rx {
            let outcome = generate(&ffmpeg, &cache_dir, &request);
            let _ = done.send(outcome);
            wake();
        }
    });
    tx
}

fn cache_path(dir: &PathBuf, key: &str) -> PathBuf {
    // 键里可能有路径字符，做个简单哈希避免非法文件名。
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in key.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    dir.join(format!("thumb-{hash:016x}.bin"))
}

fn generate(ffmpeg: &PathBuf, cache_dir: &PathBuf, request: &ThumbRequest) -> ThumbOutcome {
    let path = cache_path(cache_dir, &request.key);
    if let Ok(bytes) = std::fs::read(&path) {
        if bytes.len() >= 8 {
            let width = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let height = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            let expected = width as usize * height as usize * 4;
            if bytes.len() == 8 + expected && expected > 0 {
                return ThumbOutcome::Ready {
                    key: request.key.clone(),
                    width,
                    height,
                    rgba: bytes[8..].to_vec(),
                };
            }
        }
    }
    let video = PathBuf::from(&request.video);
    match player::capture_small(ffmpeg, &video, request.seconds, request.width, request.height) {
        Ok(rgb) => {
            let rgba = player::rgb_to_rgba(&rgb);
            let mut file = Vec::with_capacity(8 + rgba.len());
            file.extend_from_slice(&request.width.to_le_bytes());
            file.extend_from_slice(&request.height.to_le_bytes());
            file.extend_from_slice(&rgba);
            let _ = std::fs::write(&path, &file);
            ThumbOutcome::Ready { key: request.key.clone(), width: request.width, height: request.height, rgba }
        }
        Err(message) => ThumbOutcome::Failed { key: request.key.clone(), message },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_is_stable_and_unique() {
        let dir = PathBuf::from("nowhere");
        let a = cache_path(&dir, "video1-0-100");
        let b = cache_path(&dir, "video1-0-100");
        let c = cache_path(&dir, "video1-0-101");
        assert_eq!(a, b, "同键必须命中同一路径");
        assert_ne!(a, c, "不同键必须分开");
        assert!(a.file_name().unwrap().to_string_lossy().starts_with("thumb-"));
    }
}
