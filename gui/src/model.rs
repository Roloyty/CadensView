//! RPC 状态镜像与 FrameIndex 的轻量视图。

use serde::Deserialize;

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct State {
    pub video_path: Option<String>,
    pub project_path: Option<String>,
    pub dirty: bool,
    pub has_index: bool,
    pub frame_count: Option<u64>,
    pub current_frame: u64,
    pub selected_id: String,
    pub pending_start: Option<u64>,
    pub completed_count: u64,
    pub can_undo: bool,
    pub can_redo: bool,
    pub shots: Vec<ShotRow>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct ShotRow {
    pub id: String,
    pub number: usize,
    pub start: u64,
    pub end: u64,
    pub note: String,
    pub pending: bool,
    pub duration_s: f64,
}

/// Python `FrameIndex.__dict__` 中 GUI 需要的字段。
#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct IndexInfo {
    pub path: String,
    pub size: i64,
    pub mtime_ns: i64,
    pub pts: Vec<i64>,
    pub durations: Vec<i64>,
    pub time_base: String,
    pub complete: bool,
    pub metadata: IndexMetadata,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct IndexMetadata {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration_us: Option<f64>,
    pub average_rate: Option<String>,
}

impl IndexInfo {
    pub fn frames(&self) -> u64 {
        self.pts.len() as u64
    }

    /// time_base "num/den" 解析。
    pub fn time_base(&self) -> (i64, i64) {
        let mut parts = self.time_base.splitn(2, '/');
        let num: i64 = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
        let den: i64 = parts.next().and_then(|p| p.parse().ok()).unwrap_or(1);
        (num, den)
    }

    /// 帧号 → 视频内秒（ffmpeg -ss 用）。
    pub fn seconds(&self, frame: u64) -> f64 {
        let (num, den) = self.time_base();
        let pts = self.pts.get(frame as usize).copied().unwrap_or(0);
        pts as f64 * num as f64 / den as f64
    }

    pub fn size(&self) -> (usize, usize) {
        (
            self.metadata.width.unwrap_or(0) as usize,
            self.metadata.height.unwrap_or(0) as usize,
        )
    }

    /// 缩略图等磁盘缓存的稳定键：同一文件的指纹。
    pub fn cache_stem(&self) -> String {
        format!("{}-{}-{}", self.size, self.mtime_ns, self.pts.len())
    }
}

// -- 统计（stats.run 的结果视图） -------------------------------------------

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct StatsData {
    pub overview: StatsOverview,
    pub distribution: StatsDistribution,
    pub editing: StatsEditing,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct StatsOverview {
    pub shot_count: Option<u64>,
    pub scope_duration_s: Option<f64>,
    pub asl_s: Option<f64>,
    pub msl_s: Option<f64>,
    pub stddev_s: Option<f64>,
    pub cv: Option<f64>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct StatsDistribution {
    pub duration_histogram: Vec<HistBin>,
    pub percentiles_s: StatsPercentiles,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct StatsPercentiles {
    pub p10: Option<f64>,
    pub p25: Option<f64>,
    pub p75: Option<f64>,
    pub p90: Option<f64>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct HistBin {
    pub start_s: f64,
    pub end_s: f64,
    pub count: u64,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct StatsEditing {
    pub cut_density_series: Vec<DensityPoint>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct DensityPoint {
    pub video_time_s: f64,
    pub cuts_per_minute: Option<f64>,
}

// -- 色彩（palette.get / analysis.run 的结果视图） ---------------------------

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct PaletteInfo {
    pub colors: Vec<PaletteColor>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct PaletteColor {
    pub hex: String,
    pub srgb: Option<[u8; 3]>,
    pub pixel_share: Option<f64>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct BarcodeSample {
    pub video_time_s: f64,
    pub end_time_s: f64,
    pub srgb: Option<[u8; 3]>,
}

#[cfg(test)]
mod stats_tests {
    use super::*;

    #[test]
    fn stats_parse_minimal_payload() {
        let payload = serde_json::json!({
            "overview": {"shot_count": 3, "asl_s": 20.0},
            "distribution": {"duration_histogram": [
                {"start_s": 0.0, "end_s": 2.0, "count": 2, "duration_s": 5.0, "count_share": 0.5}
            ], "percentiles_s": {"p10": 1.0}},
            "editing": {"cut_density_series": [
                {"video_time_s": 0.0, "cuts_per_minute": 3.0}
            ]}
        });
        let stats: StatsData = serde_json::from_value(payload).unwrap();
        assert_eq!(stats.overview.shot_count, Some(3));
        assert_eq!(stats.distribution.duration_histogram.len(), 1);
        assert_eq!(stats.distribution.duration_histogram[0].count, 2);
        assert_eq!(stats.editing.cut_density_series.len(), 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_base_and_seconds_use_pts() {
        let index = IndexInfo {
            pts: vec![0, 24, 48, 100],
            durations: vec![1; 4],
            time_base: "1/24".into(),
            ..Default::default()
        };
        assert_eq!(index.time_base(), (1, 24));
        assert!((index.seconds(0) - 0.0).abs() < 1e-9);
        assert!((index.seconds(2) - 2.0).abs() < 1e-9);
        assert!((index.seconds(3) - 100.0 / 24.0).abs() < 1e-9);
        assert_eq!(index.frames(), 4);
        // 越界帧保守返回 0，避免播放器 seek 崩溃
        assert!((index.seconds(99) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn malformed_time_base_degrades_safely() {
        let index = IndexInfo { time_base: "oops".into(), ..Default::default() };
        assert_eq!(index.time_base(), (0, 1));
        assert!((index.seconds(0) - 0.0).abs() < 1e-9);
    }
}
