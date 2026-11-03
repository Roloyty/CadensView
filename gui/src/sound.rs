//! 声音工作区（只用 ffmpeg）：逐镜头响度/LRA/静音占比、切点处的声音、跨集音频指纹比对 OP/ED。
//! 与研究面板一样保持边车返回的 JSON，只读界面需要的字段；写操作以 Action 交回主程序。

use eframe::egui;
use serde_json::Value;
use crate::model::{self, State};
use crate::theme;

#[derive(Default)]
pub struct Sound {
    pub result: Option<Value>,
    /// 结果对应的范围/镜头键；与当前不符时主程序丢弃结果。
    pub key: String,
    pub matches: Option<Value>,
    only_jl: bool,
}

pub enum Action {
    Run,
    Match,
    AddRegion { label: String, start_frame: u64, end_frame: u64 },
    Locate(u64),
}

const FLOOR: f64 = -70.0;

fn f(value: &Value, key: &str) -> Option<f64> { value.get(key).and_then(Value::as_f64) }
fn u(value: &Value, key: &str) -> u64 { value.get(key).and_then(Value::as_u64).unwrap_or(0) }
fn s<'a>(value: &'a Value, key: &str) -> &'a str { value.get(key).and_then(Value::as_str).unwrap_or("") }
fn list<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value.get(key).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}
fn number(value: Option<f64>, digits: usize, unit: &str) -> String {
    value.map(|v| format!("{v:.digits$}{unit}")).unwrap_or_else(|| "—".into())
}
fn percent(value: Option<f64>) -> String { value.map(|v| format!("{:.0}%", v * 100.0)).unwrap_or_else(|| "—".into()) }
fn clock(seconds: f64) -> String {
    let tenths = (seconds.max(0.0) * 10.0).round() as u64;
    format!("{:02}:{:02}.{}", tenths / 600, tenths / 10 % 60, tenths % 10)
}

pub fn relation_zh(code: &str) -> &'static str {
    match code { "sync" => "同步", "J" => "J-cut 声音先行", "L" => "L-cut 声音延后", _ => "无明显变化" }
}

impl Sound {
    pub fn reset(&mut self) { *self = Sound { only_jl: self.only_jl, ..Default::default() }; }

    /// 边车结果按方法名归位；返回 true 表示项目状态可能变了（指纹写进了项目）。
    pub fn accept(&mut self, method: &str, value: Value) -> Option<bool> {
        match method {
            "audio.run" => { self.result = Some(value); Some(true) }
            "audio.match" => { self.matches = Some(value); Some(true) }
            _ => None,
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, state: &State, busy: bool) -> Vec<Action> {
        let mut actions = Vec::new();
        ui.horizontal(|ui| {
            ui.heading("声音");
            ui.weak("EBU R 128 响度 · 静音 · 切点处的声音 · 跨集音频指纹");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let label = if self.result.is_some() { "↻ 重新分析" } else { "运行声音分析" };
                if ui.add_enabled(!busy && state.completed_count > 0, egui::Button::new(label))
                    .on_hover_text("解码一遍音轨（24 分钟约数秒），范围与分析页一致").clicked() {
                    actions.push(Action::Run);
                }
                if busy { ui.spinner(); }
            });
        });
        ui.separator();
        egui::ScrollArea::vertical().id_salt("sound-body").auto_shrink([false, false]).show(ui, |ui| {
            match self.result.clone() {
                Some(result) => self.analysis(ui, &result, &mut actions),
                None => {
                    ui.weak("逐镜头响度（LUFS，按 BS.1770 门限在镜头内重新积分）、响度范围（3 秒以上镜头）、静音占比；\
                             每个切点是否落在静音里、前后 400 ms 的响度跳变，以及声音变化比画面切换早（J-cut）还是晚（L-cut）。");
                }
            }
            ui.add_space(16.0);
            self.episodes(ui, state, busy, &mut actions);
        });
        actions
    }

    fn analysis(&mut self, ui: &mut egui::Ui, result: &Value, actions: &mut Vec<Action>) {
        let summary = result.get("summary").cloned().unwrap_or(Value::Null);
        let counts = summary.get("relation_counts").cloned().unwrap_or(Value::Null);
        for warning in list(result, "warnings") {
            ui.colored_label(theme::warning(), warning.as_str().unwrap_or(""));
        }
        egui::Grid::new("sound-summary").num_columns(4).spacing([24.0, 6.0]).show(ui, |ui| {
            let cell = |ui: &mut egui::Ui, label: &str, value: String| {
                ui.label(egui::RichText::new(label).color(theme::secondary()));
                ui.monospace(value);
            };
            cell(ui, "整体响度", number(f(&summary, "integrated_lufs"), 1, " LUFS"));
            cell(ui, "响度范围 LRA", number(f(&summary, "lra_lu"), 1, " LU"));
            ui.end_row();
            cell(ui, "静音占比", percent(f(&summary, "silence_share")));
            cell(ui, "切点落在静音里", percent(f(&summary, "cuts_in_silence_share")));
            ui.end_row();
            cell(ui, "切点处响度跳变（中位）", number(f(&summary, "median_abs_loudness_jump_lu"), 1, " LU"));
            cell(ui, "声音变化偏移（中位）", number(f(&summary, "median_abs_sound_offset_s"), 2, " s"));
            ui.end_row();
        });
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new(format!("{} 个切点：", u(&summary, "cuts"))).color(theme::secondary()));
            for code in ["sync", "J", "L", "none"] {
                ui.monospace(format!("{} {}", relation_zh(code), u(&counts, code)));
            }
        });
        ui.add_space(10.0);
        ui.strong("响度曲线（瞬时响度，400 ms 窗）");
        if let Some(frame) = loudness_chart(ui, result) { actions.push(Action::Locate(frame)); }
        ui.weak("横轴为源视频时间，纵轴 −70…0 LUFS；底部短线为切镜点。点击曲线定位到该处。");
        ui.add_space(12.0);
        ui.columns(2, |columns| {
            self.shot_table(&mut columns[0], result, actions);
            self.cut_table(&mut columns[1], result, actions);
        });
        ui.add_space(6.0);
        ui.weak("J/L 判定取切点前后 1 秒内频谱变化最大的位置（变化须 ≥ 6 dB），是候选而不是结论：\
                 对白开口、音效起落也会形成变化点，研究用前请听辨核对。");
    }

    fn shot_table(&mut self, ui: &mut egui::Ui, result: &Value, actions: &mut Vec<Action>) {
        ui.strong("镜头");
        egui::ScrollArea::vertical().id_salt("sound-shots").max_height(320.0).show(ui, |ui| {
            egui::Grid::new("sound-shot-grid").striped(true).num_columns(6).show(ui, |ui| {
                for head in ["镜头", "时长", "响度", "峰值", "LRA", "静音"] { ui.weak(head); }
                ui.end_row();
                for row in list(result, "shots") {
                    if ui.link(format!("#{:03}", u(row, "shot_number"))).clicked() {
                        actions.push(Action::Locate(u(row, "start_frame")));
                    }
                    ui.monospace(number(f(row, "duration_s"), 2, "s"));
                    ui.monospace(number(f(row, "loudness_lufs"), 1, ""));
                    ui.monospace(number(f(row, "loudness_max_lufs"), 1, ""));
                    ui.monospace(number(f(row, "lra_lu"), 1, ""));
                    ui.monospace(percent(f(row, "silence_share")));
                    ui.end_row();
                }
            });
        });
    }

    fn cut_table(&mut self, ui: &mut egui::Ui, result: &Value, actions: &mut Vec<Action>) {
        ui.horizontal(|ui| {
            ui.strong("切点");
            ui.checkbox(&mut self.only_jl, "只看 J/L");
        });
        egui::ScrollArea::vertical().id_salt("sound-cuts").max_height(320.0).show(ui, |ui| {
            egui::Grid::new("sound-cut-grid").striped(true).num_columns(5).show(ui, |ui| {
                for head in ["时间", "声画关系", "偏移", "静音", "响度跳变"] { ui.weak(head); }
                ui.end_row();
                for cut in list(result, "cuts") {
                    let relation = s(cut, "relation");
                    if self.only_jl && !matches!(relation, "J" | "L") { continue; }
                    if ui.link(clock(f(cut, "time_s").unwrap_or(0.0))).on_hover_text(format!("帧 {}", u(cut, "frame"))).clicked() {
                        actions.push(Action::Locate(u(cut, "frame")));
                    }
                    if matches!(relation, "J" | "L") {
                        ui.label(egui::RichText::new(relation_zh(relation)).color(theme::accent()));
                    } else {
                        ui.label(relation_zh(relation));
                    }
                    ui.monospace(f(cut, "sound_offset_s").map(|v| format!("{v:+.2}s")).unwrap_or_else(|| "—".into()));
                    ui.label(match cut.get("in_silence").and_then(Value::as_bool) { Some(true) => "是", Some(false) => "", None => "—" });
                    ui.monospace(f(cut, "loudness_jump_lu").map(|v| format!("{v:+.1}")).unwrap_or_else(|| "—".into()));
                    ui.end_row();
                }
            });
        });
    }

    fn episodes(&mut self, ui: &mut egui::Ui, state: &State, busy: bool, actions: &mut Vec<Action>) {
        ui.heading("跨集比对 OP/ED");
        ui.weak("比对 Chromaprint 音频指纹，找出与其他集共有的长段音频（≥15 秒），按位置建议为片头曲/片尾曲区段。\
                 其他集若保存过声音分析会直接读取指纹，否则从它的源视频现算（只读，不改那个项目）。");
        ui.add_space(4.0);
        if ui.add_enabled(!busy && state.completed_count > 0, egui::Button::new("选择其他集项目…")).clicked() {
            actions.push(Action::Match);
        }
        let Some(matches) = self.matches.clone() else { return; };
        ui.add_space(6.0);
        for episode in list(&matches, "episodes") {
            if let Some(error) = episode.get("error").and_then(Value::as_str) {
                ui.colored_label(theme::error(), error);
            } else {
                ui.weak(format!("{} · {:.0} 秒 · 指纹{}{}", s(episode, "label"), f(episode, "seconds").unwrap_or(0.0),
                    if s(episode, "fingerprint") == "stored" { "取自项目" } else { "现算" },
                    match list(episode, "visual_sequences").len() { 0 => String::new(), n => format!(" · 画面共享序列 {n} 段") }));
            }
        }
        let suggestions = list(&matches, "suggestions");
        if suggestions.is_empty() {
            ui.label("没有找到 15 秒以上的共同音频。");
            return;
        }
        ui.add_space(6.0);
        for item in suggestions {
            let label = s(item, "label").to_string();
            let (start, end) = (u(item, "start_frame"), u(item, "end_frame"));
            let exists = state.regions.iter().any(|r| r.start_frame < end && start < r.end_frame);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(model::region_zh(&label)).strong());
                ui.monospace(format!("{}–{}", clock(f(item, "start_s").unwrap_or(0.0)), clock(f(item, "end_s").unwrap_or(0.0))));
                ui.weak(format!("帧 {start}–{end} · {} 集共有", list(item, "episodes").len()));
                if !list(item, "visual_support").is_empty() {
                    ui.label(egui::RichText::new("画面也重复").color(theme::accent()))
                        .on_hover_text("两边都跑过“重复镜头”时，画面上也有共享的镜头序列");
                }
                if ui.small_button("定位").clicked() { actions.push(Action::Locate(start)); }
                if ui.add_enabled(!exists, egui::Button::new("加为区段").small())
                    .on_disabled_hover_text("已与现有区段重叠").clicked() {
                    actions.push(Action::AddRegion { label: label.clone(), start_frame: start, end_frame: end });
                }
            });
        }
        ui.weak("区段边界取自指纹（约 0.12 秒一格，首尾误差约 1 秒），加入后可在“作品与区段”里核对。");
    }
}

/// 画瞬时响度曲线；点击返回对应的源帧。
fn loudness_chart(ui: &mut egui::Ui, result: &Value) -> Option<u64> {
    let curve = list(result, "curve");
    let duration = f(result, "duration_s").unwrap_or(0.0).max(0.001);
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width().max(100.0), 180.0), egui::Sense::click());
    let painter = ui.painter_at(rect);
    let plot = egui::Rect::from_min_max(rect.min + egui::vec2(44.0, 6.0), rect.max - egui::vec2(6.0, 18.0));
    let x = |t: f64| plot.left() + (t / duration).clamp(0.0, 1.0) as f32 * plot.width();
    let y = |v: f64| plot.bottom() - ((v - FLOOR) / -FLOOR).clamp(0.0, 1.0) as f32 * plot.height();
    let font = egui::FontId::monospace(10.0);
    for level in [-60.0, -40.0, -23.0, -10.0] {
        painter.line_segment([egui::pos2(plot.left(), y(level)), egui::pos2(plot.right(), y(level))],
                             egui::Stroke::new(1.0, theme::line()));
        painter.text(egui::pos2(plot.left() - 4.0, y(level)), egui::Align2::RIGHT_CENTER, format!("{level:.0}"), font.clone(), theme::faint());
    }
    for i in 0..=4 {
        let t = duration * i as f64 / 4.0;
        painter.text(egui::pos2(x(t), plot.bottom() + 3.0), egui::Align2::CENTER_TOP, clock(t), font.clone(), theme::faint());
    }
    let points: Vec<egui::Pos2> = curve.iter()
        .filter_map(|p| Some(egui::pos2(x(f(p, "video_time_s")?), y(f(p, "loudness_lufs")?)))).collect();
    for pair in points.windows(2) {
        painter.line_segment([pair[0], pair[1]], egui::Stroke::new(1.2, theme::accent()));
    }
    for cut in list(result, "cuts") {
        let cx = x(f(cut, "time_s").unwrap_or(0.0));
        let colour = if matches!(s(cut, "relation"), "J" | "L") { theme::accent() } else { theme::faint() };
        painter.line_segment([egui::pos2(cx, plot.bottom()), egui::pos2(cx, plot.bottom() - 6.0)], egui::Stroke::new(1.0, colour));
    }
    let time = response.hover_pos().filter(|p| plot.contains(*p))
        .map(|p| ((p.x - plot.left()) / plot.width()) as f64 * duration)?;
    let nearest = curve.iter().min_by(|a, b| (f(a, "video_time_s").unwrap_or(0.0) - time).abs()
        .total_cmp(&(f(b, "video_time_s").unwrap_or(0.0) - time).abs()));
    if let Some(point) = nearest {
        response.clone().on_hover_text(format!("{} · {:.1} LUFS", clock(time), f(point, "loudness_lufs").unwrap_or(FLOOR)));
    }
    if !response.clicked() { return None; }
    frame_at(result, time)
}

/// 源时间 → 帧：在所属镜头内按帧数线性插值（镜头内帧率恒定）。
fn frame_at(result: &Value, time: f64) -> Option<u64> {
    let shot = list(result, "shots").iter().find(|s| {
        f(s, "start_time_s").unwrap_or(0.0) <= time && time < f(s, "end_time_s").unwrap_or(0.0)
    })?;
    let (start, count) = (u(shot, "start_frame"), u(shot, "frame_count"));
    let share = (time - f(shot, "start_time_s")?) / f(shot, "duration_s")?.max(1e-6);
    Some(start + ((share * count as f64).floor() as u64).min(count.saturating_sub(1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn results_route_and_time_maps_to_frames() {
        let mut sound = Sound::default();
        assert_eq!(sound.accept("review.run", json!({})), None);
        assert_eq!(sound.accept("audio.run", json!({"shots": [
            {"start_frame": 0, "frame_count": 96, "start_time_s": 0.0, "end_time_s": 4.0, "duration_s": 4.0},
            {"start_frame": 96, "frame_count": 72, "start_time_s": 4.0, "end_time_s": 7.0, "duration_s": 3.0}]})), Some(true));
        let result = sound.result.clone().unwrap();
        assert_eq!(frame_at(&result, 0.0), Some(0));
        assert_eq!(frame_at(&result, 5.0), Some(120));
        assert_eq!(frame_at(&result, 9.0), None);
        sound.accept("audio.match", json!({"suggestions": []}));
        sound.reset();
        assert!(sound.result.is_none() && sound.matches.is_none());
    }

    #[test]
    fn relation_labels() {
        assert_eq!(relation_zh("J"), "J-cut 声音先行");
        assert_eq!(relation_zh("whatever"), "无明显变化");
    }
}
