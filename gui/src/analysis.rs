//! 独立分析工作区：范围由调用方固定，图表保留源时间和稳定镜头 ID。

use std::collections::HashMap;
use eframe::egui::{self, TextureHandle};
use crate::{model::*, theme};

#[derive(Clone, Copy, Default, PartialEq)]
pub enum Topic { #[default] Duration, Rhythm, Colors, Motion }

#[derive(Clone, Copy, Default, PartialEq)]
pub enum Scope { #[default] All, Marked }

#[derive(Default)]
pub struct Workspace {
    pub topic: Topic,
    pub scope: Scope,
    rolling: bool,
    detail: Option<Detail>,
}

struct Detail {
    title: String,
    description: String,
    ids: Vec<String>,
    frame: Option<u64>,
}

pub enum Action {
    RunColors,
    RunMotion,
    ExportMotion,
    ExportSlit,
    ExportStills,
    Thumbnail(u64),
    Locate { frame: u64, shot_id: Option<String> },
}

pub fn thumbnail_key(stem: &str, frame: u64) -> String { format!("{stem}-analysis-{frame}") }

impl Workspace {
    pub fn reset_results(&mut self) { self.detail = None; }

    pub fn show(&mut self, ui: &mut egui::Ui, title: &str, stats: Option<&StatsData>, colors: Option<&ColorAnalysis>,
                motion: Option<&MotionAnalysis>, thumbs: &HashMap<String, TextureHandle>, stem: &str,
                busy: bool, motion_busy: bool) -> Vec<Action> {
        let mut actions = Vec::new();
        let previous = self.topic;
        ui.horizontal(|ui| {
            for (topic, label) in [(Topic::Duration, "剪辑"), (Topic::Rhythm, "节奏"),
                                   (Topic::Colors, "色彩"), (Topic::Motion, "运动")] {
                ui.selectable_value(&mut self.topic, topic, label);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if busy { ui.spinner(); }
                if let Some(stats) = stats {
                    let scope = &stats.metadata.scope;
                    let label = if scope.kind == "selected_shots" { "已标记镜头" }
                        else if scope.analysis_complete { "全片研究" } else { "已完成部分" };
                    ui.weak(format!("{label} · {:.2}s", scope.scope_duration_s));
                }
            });
        });
        if previous != self.topic { self.detail = None; }
        ui.add_space(12.0);
        let height = ui.available_height();
        let rail_width = (ui.available_width() * 0.23).clamp(210.0, 280.0);
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(egui::vec2(rail_width, height), egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.set_max_width(rail_width);
                egui::ScrollArea::vertical().id_salt("analysis-rail").show(ui, |ui| {
                    ui.monospace(match self.topic { Topic::Duration => "01 / SHOTS", Topic::Rhythm => "02 / RHYTHM", Topic::Colors => "03 / COLOR", Topic::Motion => "04 / MOTION" });
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new(title).size(18.0).strong());
                    ui.add_space(24.0);
                    if let Some(stats) = stats {
                        self.metrics(ui, stats, colors, motion, busy, motion_busy, &mut actions);
                    } else {
                        ui.weak("完成切镜后显示指标。");
                    }
                    if self.topic == Topic::Colors {
                        ui.add_space(12.0);
                        if ui.add_enabled(!busy, egui::Button::new(if colors.is_some() { "重新计算色彩" } else { "计算色彩与代表帧" })).clicked() {
                            actions.push(Action::RunColors);
                        }
                        ui.weak("色带与代表帧采样全片；镜头色板与相邻色差使用当前范围。");
                    }
                    if let Some(stats) = stats {
                        for warning in &stats.metadata.warnings {
                            ui.add_space(6.0); ui.small(egui::RichText::new(warning).color(theme::secondary()));
                        }
                    }
                });
            });
            ui.separator();
            let width = ui.available_width();
            ui.allocate_ui_with_layout(egui::vec2(width, height), egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.set_max_width(width);
                egui::ScrollArea::vertical().id_salt("analysis-plot").show(ui, |ui| {
                    match self.topic {
                        Topic::Duration => self.duration(ui, stats),
                        Topic::Rhythm => self.rhythm(ui, stats),
                        Topic::Colors => self.colors(ui, colors, thumbs, stem, busy, &mut actions),
                        Topic::Motion => self.motion(ui, motion, busy, motion_busy, &mut actions),
                    }
                });
            });
        });
        actions
    }

    fn metrics(&mut self, ui: &mut egui::Ui, stats: &StatsData, colors: Option<&ColorAnalysis>,
               motion: Option<&MotionAnalysis>, busy: bool, motion_busy: bool, actions: &mut Vec<Action>) {
        let overview = &stats.overview;
        if self.topic == Topic::Motion {
            self.motion_metrics(ui, motion, busy, motion_busy, actions);
            return;
        }
        let primary = if self.topic == Topic::Rhythm {
            stats.special.densest_cut_windows.first().and_then(|w| w.cut_count).unwrap_or(0)
        } else { overview.shot_count.unwrap_or(0) };
        if self.topic != Topic::Colors {
            ui.weak(if self.topic == Topic::Rhythm { "最密集窗口内切镜" } else { "范围内镜头数" });
            ui.label(egui::RichText::new(format!("{primary}")).size(44.0).monospace().color(theme::accent()));
            if self.topic == Topic::Rhythm {
                if let Some(window) = stats.special.densest_cut_windows.first() {
                    ui.monospace(format!("{} — {}", clock(window.start_time_s), clock(window.end_time_s)));
                }
            }
            ui.add_space(16.0);
            metric(ui, "ASL · 平均时长", overview.asl_s, "s");
            metric(ui, "MSL · 中位时长", overview.msl_s, "s");
            metric(ui, "标准差", overview.stddev_s, "s");
            metric(ui, "变异系数", overview.cv, "");
            ui.add_space(14.0);
            for (label, shot) in [("最长镜头", overview.longest_shots.first()), ("最短镜头", overview.shortest_shots.first())] {
                if let Some(shot) = shot {
                    if ui.button(format!("{label}  #{:03} · {:.2}s", shot.shot_number, shot.duration_s)).clicked() {
                        self.detail = Some(Detail { title: label.into(), description: format!("{} 起 · {:.2}s", clock(shot.start_time_s), shot.duration_s), ids: vec![shot.shot_id.clone()], frame: Some(shot.start_frame) });
                    }
                }
            }
            if self.topic == Topic::Duration {
                ui.add_space(12.0);
                let p = &stats.distribution.percentiles_s;
                for (name, value) in [("P10", p.p10), ("P25", p.p25), ("P75", p.p75), ("P90", p.p90)] { metric(ui, name, value, "s"); }
            }
        } else if let Some(colors) = colors {
            if let Some(summary) = &colors.video_color_summary {
                ui.weak("全片主色倾向");
                ui.horizontal(|ui| {
                    for group in &summary.groups {
                        let (rect, response) = ui.allocate_exact_size(egui::vec2(58.0, 76.0), egui::Sense::hover());
                        ui.painter().rect_filled(rect, 2.0, egui::Color32::from_rgb(group.srgb[0], group.srgb[1], group.srgb[2]));
                        response.on_hover_text(format!("{} · 估计出现比例 {:.1}%", group.hex, group.coverage_share * 100.0));
                    }
                });
            }
            ui.add_space(16.0);
            let timeline = &colors.video_color_timeline;
            metric(ui, "全片平均亮度", mean(&timeline.luminance_series), "");
            metric(ui, "全片平均饱和度", mean(&timeline.saturation_series), "");
            ui.add_space(12.0);
            if ui.button("查看范围内镜头色板").clicked() {
                self.detail = Some(Detail { title: "镜头色板".into(), description: "每镜头 15% / 50% / 85% 位置采样，使用原核心的 CIELAB 色板。".into(), ids: colors.shot_color_statistics.shot_palette.iter().map(|p| p.shot.shot_id.clone()).collect(), frame: None });
            }
            ui.add_space(12.0);
            ui.weak("范围内颜色变化最大切换");
            for change in colors.shot_color_statistics.largest_color_changes.iter().take(5) {
                if ui.button(format!("{} · {:.3}", clock(change.video_time_s), change.distance)).clicked() {
                    self.detail = Some(Detail { title: "切换前后颜色差异".into(), description: format!("CIELAB 分布距离 {:.3}（0–1）；数值不表示运动或语义变化。", change.distance), ids: vec![change.left_shot_id.clone(), change.right_shot_id.clone()], frame: Some(change.boundary_frame) });
                }
            }
        }
    }

    fn duration(&mut self, ui: &mut egui::Ui, stats: Option<&StatsData>) {
        ui.heading("镜头长度");
        ui.weak("横轴：镜头时长（秒）　纵轴：镜头数；点击柱体查看对应镜头。");
        ui.add_space(16.0);
        let Some(stats) = stats else { ui.label("等待已确认镜头统计。"); return; };
        if let Some(i) = histogram(ui, &stats.distribution.duration_histogram, stats.overview.asl_s, stats.overview.msl_s) {
            let bin = &stats.distribution.duration_histogram[i];
            self.detail = Some(Detail { title: format!("{:.2} — {:.2}s", bin.start_s, bin.end_s), description: format!("{} 个镜头，时长区间左闭右开。", bin.count), ids: bin.shot_ids.clone(), frame: None });
        }
        ui.add_space(12.0);
        ui.weak("ASL 实线 · MSL 虚线。单个长镜头会抬高平均值，中位数可辅助判断。");
    }

    fn rhythm(&mut self, ui: &mut egui::Ui, stats: Option<&StatsData>) {
        ui.heading("剪辑节奏");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.rolling, false, "切镜频率 · 60s 窗口");
            ui.selectable_value(&mut self.rolling, true, "附近镜头平均时长 · 15 镜头");
        });
        let Some(stats) = stats else { ui.label("等待已确认镜头统计。"); return; };
        let duration = stats.shots.iter().map(|s| s.end_time_s).fold(0.0, f64::max)
            .max(stats.editing.cut_density_series.last().map(|p| p.video_time_s).unwrap_or(0.0));
        let points: Vec<_> = if self.rolling { stats.editing.rolling_asl_series.iter().map(|p| (p.video_time_s, p.asl_s)).collect() }
            else { stats.editing.cut_density_series.iter().map(|p| (p.video_time_s, p.cuts_per_minute)).collect() };
        let cuts: Vec<_> = stats.editing.boundaries.iter().map(|c| c.video_time_s).collect();
        if let Some(i) = line(ui, &points, duration, if self.rolling { "秒" } else { "次/分钟" }, 310.0, &cuts) {
            let (description, ids) = if self.rolling {
                let p = &stats.editing.rolling_asl_series[i];
                let first = stats.shots.iter().position(|s| s.shot_id == p.first_shot_id);
                let last = stats.shots.iter().position(|s| s.shot_id == p.last_shot_id);
                let ids = match (first, last) { (Some(a), Some(b)) if a <= b => stats.shots[a..=b].iter().map(|s| s.shot_id.clone()).collect(), _ => vec![p.shot_id.clone()] };
                (format!("{} · 附近平均 {:.2}s", clock(p.video_time_s), p.asl_s.unwrap_or(0.0)), ids)
            } else {
                let p = &stats.editing.cut_density_series[i];
                let ids = stats.shots.iter().filter(|s| s.start_time_s < p.end_time_s && s.end_time_s > p.start_time_s).map(|s| s.shot_id.clone()).collect();
                (format!("{} — {} · {} 次切换 / {:.2}s 已覆盖时长", clock(p.start_time_s), clock(p.end_time_s), p.cut_count, p.effective_window_seconds), ids)
            };
            self.detail = Some(Detail { title: "节奏窗口".into(), description, ids, frame: None });
        }
        ui.weak("保留源视频时间；未覆盖的区间留空，不把不连续选择连接成一次切镜。");
        ui.add_space(14.0);
        ui.strong("特殊节奏片段");
        for run in &stats.special.short_shot_runs {
            let label = format!("连续快切 · {} 个镜头 · {} — {}", run.shot_count, clock(run.start_time_s.unwrap_or(0.0)), clock(run.end_time_s.unwrap_or(0.0)));
            if ui.button(&label).clicked() { self.detail = Some(Detail { title: "连续快切".into(), description: label, ids: run.shot_ids.clone(), frame: None }); }
        }
        ui.add_space(10.0);
        egui::Grid::new("analysis-segments").striped(true).num_columns(4).show(ui, |ui| {
            for heading in ["源时间段", "镜头数", "ASL", "切镜/分钟"] { ui.weak(heading); } ui.end_row();
            for segment in &stats.editing.time_segment_summary {
                if ui.link(format!("{} — {}", clock(segment.start_time_s), clock(segment.end_time_s))).clicked() {
                    self.detail = Some(Detail { title: "时间分段".into(), description: "按镜头中点分配到时间段，镜头时长不做截断。".into(), ids: segment.shot_ids.clone(), frame: None });
                }
                ui.monospace(segment.shot_count.to_string()); ui.monospace(number(segment.asl_s));
                ui.monospace(number(segment.cuts_per_minute)); ui.end_row();
            }
        });
    }

    fn colors(&mut self, ui: &mut egui::Ui, colors: Option<&ColorAnalysis>, thumbs: &HashMap<String, TextureHandle>, stem: &str, busy: bool, actions: &mut Vec<Action>) {
        ui.heading("色彩序列");
        ui.weak("全片采样 · 色带按真实时间宽度绘制；点击色带或代表帧查看源画面。");
        let Some(colors) = colors else {
            ui.add_space(24.0);
            if busy { ui.label("正在计算色彩…"); }
            else if ui.button("计算色彩与代表帧").clicked() { actions.push(Action::RunColors); }
            return;
        };
        let timeline = &colors.video_color_timeline;
        if !timeline.reason.is_empty() { ui.colored_label(theme::warning(), &timeline.reason); }
        if !colors.shot_color_statistics.reason.is_empty() { ui.colored_label(theme::warning(), &colors.shot_color_statistics.reason); }
        if timeline.status == "unsupported" { return; }
        if let Some(summary) = &colors.video_color_summary {
            let count = summary.representatives.len();
            if count > 0 {
                let width = ((ui.available_width() - 8.0 * (count - 1) as f32) / count as f32).max(32.0);
                ui.horizontal_top(|ui| {
                    for still in &summary.representatives {
                        ui.vertical(|ui| {
                            if thumbnail(ui, thumbs, stem, still.source_frame, egui::vec2(width, width * 0.5625), actions) {
                                self.detail = Some(Detail { title: "全片代表帧".into(), description: format!("源时间 {} · 帧 {}", clock(still.video_time_s), still.source_frame), ids: Vec::new(), frame: Some(still.source_frame) });
                            }
                            ui.small(clock(still.video_time_s));
                        });
                    }
                });
            }
        }
        ui.add_space(16.0);
        let samples = &timeline.movie_barcode;
        let duration = samples.last().map(|s| s.end_time_s).unwrap_or(0.0).max(0.001);
        let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 150.0), egui::Sense::click());
        ui.painter().rect_filled(rect, 0.0, theme::matte());
        for sample in samples {
            if let Some([r, g, b]) = sample.srgb {
                let left = rect.left() + (sample.video_time_s / duration) as f32 * rect.width();
                let right = rect.left() + (sample.end_time_s / duration) as f32 * rect.width();
                ui.painter().rect_filled(egui::Rect::from_min_max(egui::pos2(left, rect.top()), egui::pos2(right, rect.bottom())), 0.0, egui::Color32::from_rgb(r, g, b));
            }
        }
        if let Some(pos) = response.hover_pos() {
            let seconds = ((pos.x - rect.left()) / rect.width()) as f64 * duration;
            if let Some(sample) = samples.iter().find(|s| s.video_time_s <= seconds && seconds < s.end_time_s) {
                if response.clicked() { self.detail = Some(Detail { title: "色带采样帧".into(), description: format!("{} · 帧 {}", clock(sample.video_time_s), sample.source_frame), ids: Vec::new(), frame: Some(sample.source_frame) }); }
                response.clone().on_hover_text(format!("{} · 点击查看帧 {}", clock(sample.video_time_s), sample.source_frame));
            }
        }
        ui.horizontal(|ui| { ui.monospace("00:00"); ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| { ui.monospace(clock(duration)); }); });
        for (label, points) in [("全片亮度（相对值）", &timeline.luminance_series), ("全片饱和度（相对值）", &timeline.saturation_series)] {
            ui.add_space(10.0); ui.strong(label);
            let values: Vec<_> = points.iter().map(|p| (p.video_time_s, p.value)).collect();
            if let Some(i) = line(ui, &values, duration, "0–1", 140.0, &[]) {
                self.detail = Some(Detail { title: label.into(), description: format!("{} · {:.3}", clock(points[i].video_time_s), points[i].value.unwrap_or(0.0)), ids: Vec::new(), frame: Some(points[i].source_frame) });
            }
        }
        ui.add_space(12.0);
        ui.weak(format!("当前范围：{} 个镜头色板 · {} 个相邻切换色差", colors.shot_color_statistics.shot_palette.len(), colors.shot_color_statistics.cut_color_distance.len()));
    }

    fn motion_metrics(&mut self, ui: &mut egui::Ui, motion: Option<&MotionAnalysis>,
                      busy: bool, motion_busy: bool, actions: &mut Vec<Action>) {
        let Some(motion) = motion else {
            ui.weak("逐帧比较相邻画面，得到运动量曲线与每个镜头的平均运动量。");
            ui.add_space(10.0);
            if ui.add_enabled(!busy && !motion_busy, egui::Button::new("运行运动分析")).clicked() {
                actions.push(Action::RunMotion);
            }
            ui.add_space(8.0);
            ui.weak("首次分析会解码范围内全部帧，可随时取消。");
            return;
        };
        let values: Vec<f64> = motion.shots.iter().filter_map(|s| s.motion).collect();
        let mean = if values.is_empty() { None } else { Some(values.iter().sum::<f64>() / values.len() as f64) };
        let peak = motion.shots.iter().filter_map(|s| s.motion.map(|m| (m, s)))
            .max_by(|a, b| a.0.total_cmp(&b.0));
        ui.weak("镜头平均运动量");
        ui.label(egui::RichText::new(mean.map(|v| format!("{v:.3}")).unwrap_or_else(|| "—".into()))
            .size(40.0).monospace().color(theme::accent()));
        if let Some((value, shot)) = peak {
            if ui.button(format!("最高 #{} · {:.3}", shot.shot.shot_number, value)).clicked() {
                self.detail = Some(Detail { title: "运动量最高镜头".into(),
                    description: format!("{} 起 · 平均运动量 {value:.3}", clock(shot.shot.start_time_s)),
                    ids: vec![shot.shot.shot_id.clone()], frame: Some(shot.shot.start_frame) });
            }
        }
        ui.add_space(10.0);
        metric(ui, "阈值", Some(motion.parameters.threshold as f64), "/255");
        metric(ui, "曲线采样点", Some(motion.samples.len() as f64), "");
        ui.add_space(12.0);
        if ui.add_enabled(!motion_busy, egui::Button::new("↻ 重新分析")).clicked() {
            actions.push(Action::RunMotion);
        }
        if ui.button("导出运动指纹 PNG").clicked() { actions.push(Action::ExportMotion); }
        ui.add_space(8.0);
        ui.strong("镜头运动量");
        egui::ScrollArea::vertical().id_salt("motion-shots").max_height(240.0).show(ui, |ui| {
            let mut rows: Vec<_> = motion.shots.iter().collect();
            rows.sort_by(|a, b| b.motion.unwrap_or(0.0).total_cmp(&a.motion.unwrap_or(0.0)));
            for row in rows {
                let value = row.motion.map(|v| format!("{v:.3}")).unwrap_or_else(|| "—".into());
                if ui.button(format!("#{:03}  {value}  {:.2}s", row.shot.shot_number, row.shot.duration_s)).clicked() {
                    self.detail = Some(Detail { title: format!("镜头 #{:03} 运动量", row.shot.shot_number),
                        description: format!("平均 {value} · {} 起 · {} 个样本",
                                             clock(row.shot.start_time_s), row.motion_samples),
                        ids: vec![row.shot.shot_id.clone()], frame: Some(row.shot.start_frame) });
                }
            }
        });
        ui.add_space(4.0);
        ui.weak("首帧没有同镜头内前帧，按定义排除，避免把硬切当成运动。");
    }

    fn motion(&mut self, ui: &mut egui::Ui, motion: Option<&MotionAnalysis>, busy: bool,
              motion_busy: bool, actions: &mut Vec<Action>) {
        ui.heading("运动指纹");
        ui.horizontal(|ui| {
            ui.weak("逐帧 RGB 绝对差 · 阈值 10/255");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add_enabled(!motion_busy, egui::Button::new("导出扫描条…")).clicked() {
                    actions.push(Action::ExportSlit);
                }
                if ui.add_enabled(!motion_busy, egui::Button::new("100 静帧…")).clicked() {
                    actions.push(Action::ExportStills);
                }
            });
        });
        ui.add_space(12.0);
        let Some(motion) = motion else {
            ui.label("运行运动分析后显示逐帧曲线。");
            if ui.add_enabled(!busy && !motion_busy, egui::Button::new("运行运动分析")).clicked() {
                actions.push(Action::RunMotion);
            }
            return;
        };
        let points: Vec<(f64, Option<f64>)> = motion.samples.iter()
            .map(|s| (s.video_time_s, Some(s.motion))).collect();
        let cuts: Vec<f64> = motion.shots.iter().skip(1).map(|s| s.shot.start_time_s).collect();
        if let Some(i) = line(ui, &points, motion.duration_s.max(0.001), "0–1", 360.0, &cuts) {
            let sample = &motion.samples[i];
            let shot_id = motion.shots.iter()
                .find(|s| s.shot.start_time_s <= sample.video_time_s && sample.video_time_s < s.shot.end_time_s)
                .map(|s| s.shot.shot_id.clone()).into_iter().collect();
            self.detail = Some(Detail { title: "运动采样".into(),
                description: format!("{} · 帧 {} · 运动量 {:.4}",
                                     clock(sample.video_time_s), sample.start_frame, sample.motion),
                ids: shot_id, frame: Some(sample.start_frame) });
        }
        ui.weak("横轴为源视频时间；灰线为切镜点。点击曲线查看对应帧。");
        ui.add_space(12.0);
        ui.horizontal(|ui| {
            if ui.add_enabled(!motion_busy, egui::Button::new("导出运动指纹 PNG")).clicked() {
                actions.push(Action::ExportMotion);
            }
            if ui.add_enabled(!busy && !motion_busy, egui::Button::new("重新分析")).clicked() {
                actions.push(Action::RunMotion);
            }
        });
    }

    pub fn details(&mut self, ctx: &egui::Context, stats: Option<&StatsData>, colors: Option<&ColorAnalysis>, thumbs: &HashMap<String, TextureHandle>, stem: &str) -> Vec<Action> {
        let mut actions = Vec::new();
        let Some(detail) = self.detail.as_mut() else { return actions; };
        let mut open = true;
        egui::Window::new(&detail.title).id(egui::Id::unique("analysis-detail"))
            .open(&mut open).default_width(630.0).default_height(470.0).show(ctx, |ui| {
                ui.label(&detail.description);
                if let Some(frame) = detail.frame {
                    let width = ui.available_width().min(480.0);
                    thumbnail(ui, thumbs, stem, frame, egui::vec2(width, width * 0.5625), &mut actions);
                    if ui.button(format!("在观察页定位 · 帧 {frame}")).clicked() { actions.push(Action::Locate { frame, shot_id: None }); }
                }
                ui.separator();
                egui::ScrollArea::vertical().id_salt("detail-shots").max_height(300.0).show(ui, |ui| {
                    let Some(stats) = stats else { return; };
                    for id in &detail.ids {
                        let Some(shot) = stats.shots.iter().find(|s| &s.shot_id == id) else { continue; };
                        ui.horizontal(|ui| {
                            if ui.link(format!("#{:03} · {} · {:.2}s", shot.shot_number, clock(shot.start_time_s), shot.duration_s)).clicked() { detail.frame = Some(shot.start_frame); }
                            if ui.small_button("定位").clicked() { actions.push(Action::Locate { frame: shot.start_frame, shot_id: Some(id.clone()) }); }
                            if let Some(palette) = colors.and_then(|c| c.shot_color_statistics.shot_palette.iter().find(|p| &p.shot.shot_id == id)) {
                                for color in &palette.colors {
                                    if let Some([r,g,b]) = color.srgb {
                                        let (rect, response) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
                                        ui.painter().rect_filled(rect, 1.0, egui::Color32::from_rgb(r,g,b));
                                        response.on_hover_text(format!("{} · {:.1}%", color.hex, color.pixel_share.unwrap_or(0.0) * 100.0));
                                    }
                                }
                            }
                        });
                    }
                    if detail.ids.is_empty() && detail.frame.is_none() { ui.weak("此范围没有镜头。"); }
                });
            });
        if !open || ctx.input(|i| i.key_pressed(egui::Key::Escape)) { self.detail = None; }
        actions
    }
}

fn metric(ui: &mut egui::Ui, label: &str, value: Option<f64>, unit: &str) {
    ui.horizontal(|ui| { ui.label(egui::RichText::new(label).color(theme::secondary()));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| { ui.monospace(format!("{} {unit}", number(value))); }); });
}
fn number(value: Option<f64>) -> String { value.filter(|v| v.is_finite()).map(|v| format!("{v:.2}")).unwrap_or_else(|| "—".into()) }
fn clock(seconds: f64) -> String { let seconds = seconds.max(0.0) as u64; format!("{:02}:{:02}", seconds / 60, seconds % 60) }
fn mean(points: &[ColorPoint]) -> Option<f64> { let values: Vec<_> = points.iter().filter_map(|p| p.value).collect(); if values.is_empty() { None } else { Some(values.iter().sum::<f64>() / values.len() as f64) } }

fn thumbnail(ui: &mut egui::Ui, thumbs: &HashMap<String, TextureHandle>, stem: &str, frame: u64, size: egui::Vec2, actions: &mut Vec<Action>) -> bool {
    let key = thumbnail_key(stem, frame);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    ui.painter().rect_filled(rect, 0.0, theme::matte());
    if let Some(texture) = thumbs.get(&key) {
        let image = texture.size_vec2();
        let fitted = image * (size.x / image.x).min(size.y / image.y);
        ui.painter().image(texture.id(), egui::Rect::from_center_size(rect.center(), fitted), egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
    } else {
        actions.push(Action::Thumbnail(frame));
        ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, format!("帧 {frame} …"), egui::FontId::monospace(12.0), theme::secondary());
    }
    response.clicked()
}

fn axes(ui: &egui::Ui, rect: egui::Rect, max_x: f64, max_y: f64, time: bool, unit: &str) -> egui::Rect {
    let plot = egui::Rect::from_min_max(rect.min + egui::vec2(50.0, 24.0), rect.max - egui::vec2(14.0, 32.0));
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, theme::input());
    for i in 0..=4 {
        let fraction = i as f32 / 4.0;
        let y = plot.bottom() - plot.height() * fraction;
        painter.line_segment([egui::pos2(plot.left(), y), egui::pos2(plot.right(), y)], egui::Stroke::new(0.5, theme::line()));
        painter.text(egui::pos2(plot.left() - 8.0, y), egui::Align2::RIGHT_CENTER, format!("{:.1}", max_y * fraction as f64), egui::FontId::monospace(10.0), theme::faint());
        let x = plot.left() + plot.width() * fraction;
        let value = max_x * fraction as f64;
        painter.text(egui::pos2(x, plot.bottom() + 10.0), egui::Align2::CENTER_TOP, if time { clock(value) } else { format!("{value:.1}s") }, egui::FontId::monospace(10.0), theme::faint());
    }
    painter.text(rect.min + egui::vec2(50.0, 6.0), egui::Align2::LEFT_TOP, unit, egui::FontId::proportional(10.0), theme::secondary());
    plot
}

fn histogram(ui: &mut egui::Ui, bins: &[HistBin], asl: Option<f64>, msl: Option<f64>) -> Option<usize> {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width().max(100.0), 330.0), egui::Sense::click());
    let max_x = bins.last().map(|b| b.end_s).unwrap_or(1.0).max(0.001);
    let max_y = bins.iter().map(|b| b.count as f64).fold(1.0, f64::max);
    let plot = axes(ui, rect, max_x, max_y, false, "镜头数");
    let hovered = response.hover_pos().filter(|p| plot.contains(*p)).and_then(|p| bin_at(bins, ((p.x - plot.left()) / plot.width()) as f64 * max_x));
    for (i, bin) in bins.iter().enumerate() {
        let left = plot.left() + (bin.start_s / max_x) as f32 * plot.width();
        let right = plot.left() + (bin.end_s / max_x) as f32 * plot.width();
        let top = plot.bottom() - (bin.count as f64 / max_y) as f32 * plot.height();
        ui.painter().rect_filled(egui::Rect::from_min_max(egui::pos2(left + 1.0, top), egui::pos2((right - 1.0).max(left + 1.0), plot.bottom())), 0.0,
            if hovered == Some(i) { theme::selected() } else { theme::accent().gamma_multiply(0.75) });
    }
    for (i, (label, value)) in [("ASL", asl), ("MSL", msl)].into_iter().enumerate() {
        if let Some(value) = value {
            let x = plot.left() + (value / max_x) as f32 * plot.width();
            if i == 0 { ui.painter().line_segment([egui::pos2(x, plot.top()), egui::pos2(x, plot.bottom())], egui::Stroke::new(1.0, theme::ink())); }
            else { let mut y = plot.top(); while y < plot.bottom() { ui.painter().line_segment([egui::pos2(x, y), egui::pos2(x, (y+3.0).min(plot.bottom()))], egui::Stroke::new(1.0, theme::secondary())); y += 7.0; } }
            ui.painter().text(egui::pos2(x + 4.0, plot.top() + 8.0 + i as f32 * 16.0), egui::Align2::LEFT_TOP, label, egui::FontId::monospace(11.0), theme::ink());
        }
    }
    if let Some(i) = hovered { let bin = &bins[i]; response.clone().on_hover_text(format!("{:.2}–{:.2}s · {} 个镜头", bin.start_s, bin.end_s, bin.count)); }
    if response.clicked() { hovered } else { None }
}

fn bin_at(bins: &[HistBin], value: f64) -> Option<usize> { bins.iter().position(|bin| bin.start_s <= value && value < bin.end_s) }

fn line(ui: &mut egui::Ui, points: &[(f64, Option<f64>)], duration: f64, unit: &str, height: f32, cuts: &[f64]) -> Option<usize> {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width().max(100.0), height), egui::Sense::click());
    let max_x = duration.max(0.001);
    let max_y = points.iter().filter_map(|p| p.1).fold(0.01, f64::max);
    let plot = axes(ui, rect, max_x, max_y, true, unit);
    let position = |time: f64, value: f64| egui::pos2(plot.left() + (time / max_x) as f32 * plot.width(), plot.bottom() - (value / max_y) as f32 * plot.height());
    let mut previous = None;
    for (time, value) in points {
        if let Some(value) = value.filter(|v| v.is_finite()) {
            let pos = position(*time, value);
            if let Some(old) = previous { ui.painter().line_segment([old, pos], egui::Stroke::new(1.5, theme::accent())); }
            else { ui.painter().circle_filled(pos, 2.0, theme::accent()); }
            previous = Some(pos);
        } else { previous = None; }
    }
    for cut in cuts { let x = position(*cut, 0.0).x; ui.painter().line_segment([egui::pos2(x, plot.bottom()), egui::pos2(x, plot.bottom()-5.0)], egui::Stroke::new(1.0, theme::faint())); }
    let hovered = response.hover_pos().filter(|p| plot.contains(*p)).and_then(|p| {
        let time = ((p.x - plot.left()) / plot.width()) as f64 * max_x;
        points.iter().enumerate().filter(|(_, p)| p.1.is_some()).min_by(|a,b| (a.1.0-time).abs().total_cmp(&(b.1.0-time).abs())).map(|(i,_)| i)
    });
    if let Some(i) = hovered {
        let p = points[i]; let pos = position(p.0, p.1.unwrap());
        ui.painter().circle_filled(pos, 3.5, theme::selected());
        response.clone().on_hover_text(format!("{} · {} {unit}", clock(p.0), number(p.1)));
    }
    if response.clicked() { hovered } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payloads() -> (StatsData, ColorAnalysis, MotionAnalysis) {
        let stats = serde_json::from_value(serde_json::json!({
            "metadata": {"scope": {"kind": "selected_shots", "analysis_complete": true, "scope_duration_s": 4.0}},
            "shots": [{"shot_id": "a", "shot_number": 1, "start_frame": 0, "end_frame_exclusive": 24,
                       "start_time_s": 0.0, "end_time_s": 1.0, "duration_s": 1.0},
                      {"shot_id": "b", "shot_number": 2, "start_frame": 48, "end_frame_exclusive": 96,
                       "start_time_s": 2.0, "end_time_s": 4.0, "duration_s": 2.0}],
            "overview": {"shot_count": 2, "asl_s": 1.5, "msl_s": 1.5,
                         "longest_shots": [{"shot_id": "b", "shot_number": 2, "duration_s": 2.0,
                                            "start_time_s": 2.0, "start_frame": 48}],
                         "shortest_shots": [{"shot_id": "a", "shot_number": 1, "duration_s": 1.0,
                                             "start_time_s": 0.0, "start_frame": 0}]},
            "distribution": {"duration_histogram": [
                {"start_s": 0.0, "end_s": 2.0, "count": 1, "shot_ids": ["a"]},
                {"start_s": 2.0, "end_s": 4.0, "count": 1, "shot_ids": ["b"]}],
                "percentiles_s": {"p10": 1.0, "p50": 1.5}},
            "editing": {"cut_density_series": [
                {"video_time_s": 1.0, "start_time_s": 0.5, "end_time_s": 1.5,
                 "cuts_per_minute": 0.0, "effective_window_seconds": 1.0, "cut_count": 0}],
                "rolling_asl_series": [{"video_time_s": 0.5, "asl_s": 1.0, "shot_id": "a",
                                        "first_shot_id": "a", "last_shot_id": "a"}],
                "time_segment_summary": [{"start_time_s": 0.0, "end_time_s": 4.0, "shot_count": 2,
                                          "asl_s": 1.5, "cuts_per_minute": 0.0, "shot_ids": ["a", "b"]}],
                "boundaries": []},
            "special": {"densest_cut_windows": [], "short_shot_runs": []}
        })).unwrap();
        let colors = serde_json::from_value(serde_json::json!({
            "video_color_timeline": {"movie_barcode": [
                {"video_time_s": 0.0, "end_time_s": 1.0, "srgb": [255, 0, 0], "source_frame": 0},
                {"video_time_s": 1.0, "end_time_s": 2.0, "srgb": [0, 0, 255], "source_frame": 24}],
                "luminance_series": [{"video_time_s": 0.0, "value": 0.5, "source_frame": 0}],
                "saturation_series": [{"video_time_s": 0.0, "value": 0.7, "source_frame": 0}],
                "status": "ok", "reason": ""},
            "video_color_summary": {"groups": [{"srgb": [200, 10, 10], "hex": "#c80a0a", "coverage_share": 0.6}],
                                    "representatives": [{"source_frame": 0, "video_time_s": 0.0}]},
            "shot_color_statistics": {"shot_palette": [], "cut_color_distance": [],
                                      "largest_color_changes": [], "status": "ok", "reason": ""}
        })).unwrap();
        let motion = serde_json::from_value(serde_json::json!({
            "duration_s": 4.0,
            "parameters": {"threshold": 10, "first_frame": "excluded"},
            "samples": [
                {"start_frame": 0, "end_frame": 24, "video_time_s": 0.0, "motion": 0.0},
                {"start_frame": 24, "end_frame": 48, "video_time_s": 1.0, "motion": 0.2}
            ],
            "shots": [
                {"shot_id": "a", "shot_number": 1, "start_frame": 0, "end_frame_exclusive": 24,
                 "start_time_s": 0.0, "end_time_s": 1.0, "duration_s": 1.0, "motion": 0.0, "motion_samples": 23},
                {"shot_id": "b", "shot_number": 2, "start_frame": 24, "end_frame_exclusive": 96,
                 "start_time_s": 1.0, "end_time_s": 4.0, "duration_s": 3.0, "motion": 0.2, "motion_samples": 71}
            ]
        })).unwrap();
        (stats, colors, motion)
    }

    #[test]
    fn all_topics_render_with_scoped_payload_without_panicking() {
        let (stats, colors, motion) = payloads();
        let thumbs = HashMap::new();
        let mut workspace = Workspace::default();
        for topic in [Topic::Duration, Topic::Rhythm, Topic::Colors, Topic::Motion] {
            workspace.topic = topic;
            let ctx = egui::Context::default();
            let mut output = ctx.run_ui(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let actions = workspace.show(ui, "clip", Some(&stats), Some(&colors), Some(&motion), &thumbs, "stem", false, false);
                    assert!(actions.len() <= 1);
                });
            });
            output.textures_delta.clear();
        }
    }

    #[test]
    fn histogram_hit_test_uses_actual_half_open_bins() {
        let bins = vec![HistBin { start_s: 0.0, end_s: 2.0, ..Default::default() }, HistBin { start_s: 2.0, end_s: 5.0, ..Default::default() }];
        assert_eq!(bin_at(&bins, 0.0), Some(0)); assert_eq!(bin_at(&bins, 2.0), Some(1));
        assert_eq!(bin_at(&bins, 4.9), Some(1)); assert_eq!(bin_at(&bins, 5.0), None);
    }
}
