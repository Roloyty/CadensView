//! RPC 状态镜像与 FrameIndex 的轻量视图。

use serde::Deserialize;

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct State {
    #[serde(default, deserialize_with = "string_or_null")]
    pub project_id: String,
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
    pub regions: Vec<Region>,
    pub metadata: std::collections::BTreeMap<String, String>,
    pub has_detection_record: bool,
    pub review_confirmed: u64,
}

/// 内容区段（OP/ED/预告…），统计时可整体排除。
#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct Region {
    pub id: String,
    pub label: String,
    pub start_frame: u64,
    pub end_frame: u64,
}

pub const REGION_LABELS: [(&str, &str); 7] = [
    ("OP", "片头曲"), ("ED", "片尾曲"), ("preview", "下集预告"), ("recap", "前情回顾"),
    ("eyecatch", "眼动画/幕间"), ("title", "标题卡"), ("other", "其他"),
];

/// 景别代码 → 中文；人工标注、模型、VLM 共用同一套五类。
pub const SCALES: [(&str, &str); 5] = [("ECS", "大特写"), ("CS", "特写"), ("MS", "中景"), ("FS", "全景"), ("LS", "远景")];

pub fn scale_zh(code: &str) -> &str {
    SCALES.iter().find(|(c, _)| *c == code).map(|(_, zh)| *zh).unwrap_or(code)
}

pub fn region_zh(code: &str) -> &str {
    REGION_LABELS.iter().find(|(c, _)| *c == code).map(|(_, zh)| *zh).unwrap_or(code)
}

/// VLM 推断层（与人工标注分开显示）。
#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct VlmInfo {
    pub shot_scale: Option<String>,
    pub camera_movement: Option<String>,
    pub camera_angle: Option<String>,
    pub content_type: Option<String>,
    pub special_segment: Option<String>,
    pub effects: Vec<String>,
    pub description: String,
    pub confidence: Option<f64>,
    pub model: String,
}

fn string_or_null<'de, D>(deserializer: D) -> Result<String, D::Error>
where D: serde::Deserializer<'de> {
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
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
    pub tags: Vec<String>,
    pub scale: Option<ScaleInfo>,
    pub scale_manual: Option<String>,
    pub cut_number: String,
    pub vlm: Option<VlmInfo>,
    pub region: Option<String>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct ScaleInfo {
    pub label: String,
    pub label_zh: String,
    pub confidence: Option<f64>,
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
    pub audio_tracks: Option<u32>,
}

impl IndexInfo {
    pub fn frames(&self) -> u64 {
        self.pts.len() as u64
    }

    pub fn time_base(&self) -> (i64, i64) {
        let mut parts = self.time_base.splitn(2, '/');
        let num = parts.next().and_then(|p| p.parse::<i64>().ok()).unwrap_or(0);
        let den = parts.next().and_then(|p| p.parse::<i64>().ok()).filter(|d| *d > 0).unwrap_or(1);
        (num, den)
    }

    /// 帧号 → 相对首帧的秒数，与统计核心的 source-time 坐标一致。
    pub fn seconds(&self, frame: u64) -> f64 {
        let (num, den) = self.time_base();
        match (self.pts.get(frame as usize), self.pts.first()) {
            (Some(pts), Some(origin)) => (*pts - *origin) as f64 * num as f64 / den as f64,
            _ => 0.0,
        }
    }

    /// 最后显示帧的排他终点；循环音频和总时长不能截去最后一帧。
    pub fn endpoint_seconds(&self, frame: u64) -> f64 {
        if frame < self.frames() {
            return self.seconds(frame);
        }
        let last = self.frames().saturating_sub(1);
        let (num, den) = self.time_base();
        let duration = self.durations.last().copied().filter(|v| *v > 0)
            .or_else(|| self.pts.windows(2).last().map(|p| p[1] - p[0])).unwrap_or(0);
        self.seconds(last) + duration as f64 * num as f64 / den as f64
    }

    pub fn frame_at(&self, seconds: f64) -> u64 {
        if !seconds.is_finite() || seconds < 0.0 || self.pts.is_empty() {
            return 0;
        }
        let (num, den) = self.time_base();
        if num <= 0 { return 0; }
        let target = self.pts[0] as f64 + seconds * den as f64 / num as f64;
        let after = self.pts.partition_point(|pts| (*pts as f64) <= target);
        after.saturating_sub(1).min(self.pts.len() - 1) as u64
    }

    pub fn size(&self) -> (usize, usize) {
        (self.metadata.width.unwrap_or(0) as usize, self.metadata.height.unwrap_or(0) as usize)
    }

    pub fn cache_stem(&self) -> String {
        format!("{}-{}-{}-{}", self.path, self.size, self.mtime_ns, self.pts.len())
    }
}

// -- 统计（保留实际 JSON 的范围、镜头 ID 与时间窗，用于图表回看） ------------

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct StatsData {
    pub metadata: AnalysisMetadata,
    pub shots: Vec<ShotMetric>,
    pub overview: StatsOverview,
    pub distribution: StatsDistribution,
    pub editing: StatsEditing,
    pub special: StatsSpecial,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct AnalysisMetadata {
    pub scope: ScopeInfo,
    pub warnings: Vec<String>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct ScopeInfo {
    pub kind: String,
    pub analysis_complete: bool,
    pub scope_duration_s: f64,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct ShotMetric {
    pub shot_id: String,
    pub shot_number: usize,
    pub start_frame: u64,
    pub end_frame_exclusive: u64,
    pub start_time_s: f64,
    pub end_time_s: f64,
    pub duration_s: f64,
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
    pub longest_shots: Vec<ShotMetric>,
    pub shortest_shots: Vec<ShotMetric>,
    pub msl_ci95_s: Option<[f64; 2]>,
    pub iqr_s: Option<f64>,
    pub cutting_rate_per_min: Option<f64>,
    pub cutting_rate_ci95_per_min: Option<[f64; 2]>,
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
    pub shot_ids: Vec<String>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct StatsEditing {
    pub cut_density_series: Vec<DensityPoint>,
    pub rolling_asl_series: Vec<RollingPoint>,
    pub time_segment_summary: Vec<SegmentRow>,
    pub boundaries: Vec<CutBoundary>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct CutBoundary {
    pub video_time_s: f64,
    pub left_shot_id: String,
    pub right_shot_id: String,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct RollingPoint {
    pub video_time_s: f64,
    pub asl_s: Option<f64>,
    pub shot_id: String,
    pub first_shot_id: String,
    pub last_shot_id: String,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct SegmentRow {
    pub start_time_s: f64,
    pub end_time_s: f64,
    pub shot_count: u64,
    pub asl_s: Option<f64>,
    pub cuts_per_minute: Option<f64>,
    pub shot_ids: Vec<String>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct StatsSpecial {
    pub densest_cut_windows: Vec<DensestWindow>,
    pub short_shot_runs: Vec<ShortRun>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct DensestWindow {
    pub start_time_s: f64,
    pub end_time_s: f64,
    pub cut_count: Option<u64>,
    pub cuts_per_minute: Option<f64>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct ShortRun {
    pub start_time_s: Option<f64>,
    pub end_time_s: Option<f64>,
    pub shot_count: u64,
    pub shot_ids: Vec<String>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct DensityPoint {
    pub video_time_s: f64,
    pub start_time_s: f64,
    pub end_time_s: f64,
    pub cuts_per_minute: Option<f64>,
    pub effective_window_seconds: f64,
    pub cut_count: u64,
}

// -- 色彩（全片时间轴与有范围的镜头色板分开保留） -----------------------------

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
    pub source_frame: u64,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct ColorAnalysis {
    pub video_color_timeline: ColorTimeline,
    pub video_color_summary: Option<ColorSummary>,
    pub shot_color_statistics: ShotColors,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct ColorTimeline {
    pub movie_barcode: Vec<BarcodeSample>,
    pub luminance_series: Vec<ColorPoint>,
    pub saturation_series: Vec<ColorPoint>,
    pub status: String,
    pub reason: String,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct ColorPoint {
    pub video_time_s: f64,
    pub value: Option<f64>,
    pub source_frame: u64,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct ColorSummary {
    pub groups: Vec<ColorGroup>,
    pub representatives: Vec<Representative>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct ColorGroup {
    pub srgb: [u8; 3],
    pub hex: String,
    pub coverage_share: f64,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct Representative {
    pub source_frame: u64,
    pub video_time_s: f64,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct ShotColors {
    pub shot_palette: Vec<ShotPalette>,
    pub cut_color_distance: Vec<ColorChange>,
    pub largest_color_changes: Vec<ColorChange>,
    pub status: String,
    pub reason: String,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct ShotPalette {
    #[serde(flatten)]
    pub shot: ShotMetric,
    pub colors: Vec<PaletteColor>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct ColorChange {
    pub video_time_s: f64,
    pub boundary_frame: u64,
    pub left_shot_id: String,
    pub right_shot_id: String,
    pub distance: f64,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct MotionAnalysis {
    pub samples: Vec<MotionSample>,
    pub shots: Vec<MotionShot>,
    pub duration_s: f64,
    pub parameters: MotionParameters,
    pub timing_summary: TimingSummary,
    pub warnings: Vec<String>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct TimingSummary {
    pub shots_with_reliable_koma: u64,
    pub koma_shot_counts: std::collections::BTreeMap<String, u64>,
    pub mean_drawings_per_s: Option<f64>,
    pub mean_still_share: Option<f64>,
    pub mean_pan_share: Option<f64>,
    pub koma_changing_shots: u64,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct KomaSegment {
    pub start_frame: u64,
    pub end_frame: u64,
    pub koma_label: String,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct MotionSample {
    pub start_frame: u64,
    pub end_frame: u64,
    pub video_time_s: f64,
    pub motion: f64,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct MotionShot {
    #[serde(flatten)]
    pub shot: ShotMetric,
    pub motion: Option<f64>,
    pub motion_samples: u64,
    pub koma_label: Option<String>,
    pub koma_reliable: bool,
    pub drawings_per_s: Option<f64>,
    pub still_share: Option<f64>,
    pub pan_share: Option<f64>,
    pub drawn_motion: Option<f64>,
    pub koma_segments: Vec<KomaSegment>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct MotionParameters {
    pub threshold: u32,
    pub first_frame: String,
    pub definition: String,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct SubtitleCue {
    pub start_us: i64,
    pub end_us: i64,
    pub text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_coordinates_use_nonzero_origin_and_exclusive_endpoint() {
        let index = IndexInfo {
            pts: vec![240, 264, 288, 340], durations: vec![24; 4],
            time_base: "1/24".into(), ..Default::default()
        };
        assert_eq!(index.seconds(0), 0.0);
        assert_eq!(index.seconds(2), 2.0);
        assert!((index.endpoint_seconds(4) - 124.0 / 24.0).abs() < 1e-9);
        assert_eq!(index.frame_at(1.8), 1);
        assert_eq!(index.frame_at(999.0), 3);
        assert_eq!(index.seconds(99), 0.0);
    }

    #[test]
    fn null_project_id_from_rpc_is_empty_not_a_parse_failure() {
        let state: State = serde_json::from_value(serde_json::json!({
            "project_id": null, "dirty": false, "shots": []
        })).unwrap();
        assert_eq!(state.project_id, "");
        let state: State = serde_json::from_value(serde_json::json!({"project_id": "abc"})).unwrap();
        assert_eq!(state.project_id, "abc");
    }

    #[test]
    fn research_layers_survive_rpc_payload() {
        let state: State = serde_json::from_value(serde_json::json!({
            "project_id": "p", "regions": [{"id": "r1", "label": "OP", "start_frame": 0, "end_frame": 120}],
            "metadata": {"title": "片"}, "has_detection_record": true,
            "shots": [{"id": "a", "number": 0, "start": 0, "end": 24, "scale_manual": "CS", "cut_number": "012",
                       "region": "OP", "vlm": {"shot_scale": "MS", "camera_movement": "pan", "effects": ["glow"],
                       "description": "对话", "confidence": 0.8}}]
        })).unwrap();
        assert_eq!(state.regions[0].label, "OP");
        assert_eq!(state.metadata["title"], "片");
        let shot = &state.shots[0];
        assert_eq!((shot.scale_manual.as_deref(), shot.cut_number.as_str()), (Some("CS"), "012"));
        assert_eq!(shot.vlm.as_ref().unwrap().shot_scale.as_deref(), Some("MS"));
        assert_eq!(scale_zh("ECS"), "大特写");
        assert_eq!(region_zh("ED"), "片尾曲");
    }

    #[test]
    fn motion_timing_and_robust_overview_parse() {
        let motion: MotionAnalysis = serde_json::from_value(serde_json::json!({
            "shots": [{"shot_id": "a", "shot_number": 1, "start_frame": 0, "end_frame_exclusive": 36,
                       "motion": 0.1, "koma_label": "一拍三", "koma_reliable": true, "drawings_per_s": 8.0,
                       "koma_segments": [{"start_frame": 0, "end_frame": 12, "koma_label": "一拍三"}]}],
            "timing_summary": {"shots_with_reliable_koma": 1, "koma_shot_counts": {"一拍三": 1}},
            "warnings": ["IVTC"]
        })).unwrap();
        assert_eq!(motion.shots[0].koma_label.as_deref(), Some("一拍三"));
        assert_eq!(motion.timing_summary.koma_shot_counts["一拍三"], 1);
        let overview: StatsOverview = serde_json::from_value(serde_json::json!({
            "msl_s": 2.0, "msl_ci95_s": [1.5, 2.5], "cutting_rate_per_min": 30.0,
            "cutting_rate_ci95_per_min": null
        })).unwrap();
        assert_eq!(overview.msl_ci95_s, Some([1.5, 2.5]));
        assert!(overview.cutting_rate_ci95_per_min.is_none());
    }

    #[test]
    fn malformed_time_base_degrades_safely() {
        for value in ["oops", "1/0", "1/-1"] {
            let index = IndexInfo { time_base: value.into(), ..Default::default() };
            assert_eq!(index.time_base().1, 1);
            assert_eq!(index.seconds(0), 0.0);
        }
    }

    #[test]
    fn scope_and_drilldown_ids_survive_rpc_payload() {
        let value = serde_json::json!({
            "metadata": {"scope": {"kind": "selected_shots", "analysis_complete": true}},
            "shots": [{"shot_id": "one", "shot_number": 1, "start_frame": 0,
                "end_frame_exclusive": 24, "duration_s": 1.0}],
            "overview": {"shot_count": 1, "asl_s": 1.0},
            "distribution": {"duration_histogram": [
                {"start_s": 0.0, "end_s": 2.0, "count": 1, "shot_ids": ["one"]}
            ]},
            "editing": {"rolling_asl_series": [{"video_time_s": 0.5, "asl_s": 1.0,
                "first_shot_id": "one", "last_shot_id": "one", "shot_id": "one"}]}
        });
        let stats: StatsData = serde_json::from_value(value).unwrap();
        assert_eq!(stats.metadata.scope.kind, "selected_shots");
        assert_eq!(stats.distribution.duration_histogram[0].shot_ids, ["one"]);
        assert_eq!(stats.shots[0].end_frame_exclusive, 24);
        assert_eq!(stats.editing.rolling_asl_series[0].shot_id, "one");
    }
}
