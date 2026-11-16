//! Source-anchored annotation, editable exposure sheets and audio/video review.
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, TextureHandle, Vec2};
use serde::Deserialize;
use serde_json::{json, Value};
use crate::{analysis, model, model::{IndexInfo, State}, theme};

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct Data {
    pub tiers: Vec<Tier>,
    pub events: Vec<Event>,
    pub exposures: Vec<Exposure>,
    pub sound_reviews: BTreeMap<String, SoundReview>,
    #[serde(default)]
    pub version: u64,
    #[serde(default)]
    pub provenance: Value,
    #[serde(default)]
    pub shot_relations: Vec<Value>,
    #[serde(default)]
    pub evidence_cards: Vec<Value>,
    #[serde(default)]
    pub adjudications: Vec<Value>,
    #[serde(default)]
    pub speaker_links: Vec<Value>,
    #[serde(default)]
    pub color_observations: Vec<Value>,
    #[serde(default)]
    pub version_alignments: Vec<Value>,
    #[serde(default)]
    pub saved_queries: Vec<Value>,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct Tier { pub id: String, pub name: String, pub kind: String, pub vocabulary: Vec<String> }

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct Event {
    pub id: String, pub tier_id: String, pub start_frame: u64, pub end_frame: u64,
    pub label: String, pub speaker: String, pub note: String, pub author: String,
    pub source: String, pub review_status: String,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct Exposure {
    pub id: String, pub name: String, pub start_frame: u64, pub end_frame: u64,
    pub roi: [f32; 4], pub boundaries: Vec<u64>, pub estimated_boundaries: Vec<u64>,
    pub changes: Vec<f32>, pub review_status: String, pub warning: Option<String>,
    #[serde(default)]
    pub phases: Vec<Value>,
    #[serde(default)]
    pub phase_template: String,
}

#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default)]
pub struct SoundReview { pub signature: Vec<Value>, pub relation: String }

#[derive(Default, PartialEq)]
enum Tab {
    #[default]
    Timeline,
    Exposure,
    Dialogue,
    Relations,
    Evidence,
    SoundLinks,
    Color,
    Query,
}

pub enum Action {
    Rpc(&'static str, Value),
    Job(&'static str, Value, &'static str),
    Seek(u64), Loop(u64, u64), Stop, Thumbnail(u64), LoadSubtitles, Export,
    ImportJson,
    ImportEaf,
    ExportFrames(String),
    AttachEvidence(u64),
    Value(&'static str, Value),
}

const RELATIONS: [(&str, &str); 10] = [
    ("shot_reverse", "正打/反打"), ("reaction", "反应镜头"), ("eyeline", "视线匹配"),
    ("action_match", "动作接戏"), ("graphic_match", "图形匹配"), ("insert", "插入镜头"),
    ("parallel", "平行剪辑"), ("motif", "重复母题"), ("transition", "转场"), ("other", "其他"),
];
const PHASE_TEMPLATES: [(&str, &str); 3] = [
    ("classic", "预备 / 动作 / 缓冲 / 收势"), ("impact", "预备 / 动作 / 保持"),
    ("minimal", "保持 / 动作 / 保持"),
];
const CARD_KINDS: [(&str, &str); 2] = [("documented", "记录 / 出处"), ("interpretation", "判断 / 解释")];
const QUERY_FIELDS: [(&str, &str); 10] = [
    ("text", "文字"), ("tag", "标签"), ("scale", "景别"), ("camera", "运镜"),
    ("duration_min", "最短时长"), ("duration_max", "最长时长"), ("stillness_min", "静止率下限"),
    ("stillness_max", "静止率上限"), ("metadata", "作品信息"), ("kind", "时间层类型"),
];

pub struct Workspace {
    project: String,
    tab: Tab,
    pub wave: Option<Value>,
    start: u64, end: u64,
    tier_id: String, tier_name: String, tier_kind: String, vocabulary: String,
    editing: Event,
    author: String,
    query: String,
    sheet_id: String, sheet_name: String,
    roi: [f32; 4], roi_drag: Option<Pos2>, roi_pick: bool, onion: bool,
    relation_type: String, transition_type: String, source_shot: String, target_shot: String,
    explanation: String, relation_status: String,
    evidence_title: String, evidence_kind: String, documented: String,
    interpretation: String, bibliography: String, attachment: Option<(u64, String)>,
    card_status: String, decision: String, decision_reason: String,
    speaker: String, character: String, speaker_position: String, speaker_note: String,
    color_label: String, color_scene: String, color_phase: String, color_gaze: String,
    color_judgment: String, subject: [f32; 4], exclusions: Vec<[f32; 4]>,
    color_result: Option<Value>,
    query_name: String, query_field: String, query_value: String, query_result: Option<Value>,
    query_clauses: Vec<Value>, query_match_any: bool,
    asr_adapter: String, asr_status: Option<Value>, no_dialogue: Option<Value>,
    asr_segments_json: String, asr_model: String, asr_model_version: String, asr_source: String,
    relation_report: Option<Value>, evidence_report: Option<Value>,
    difference_note: String,
}

impl Default for Workspace {
    fn default() -> Self {
        Self { project: String::new(), tab: Tab::default(), wave: None, start: 0, end: 0,
            tier_id: String::new(), tier_name: String::new(), tier_kind: "custom".into(), vocabulary: String::new(),
            editing: Event::default(), author: String::new(), query: String::new(), sheet_id: String::new(),
            sheet_name: "角色曝光".into(), roi: [0.0, 0.0, 1.0, 1.0], roi_drag: None, roi_pick: false, onion: false,
            relation_type: "shot_reverse".into(), transition_type: "cut".into(), source_shot: String::new(),
            target_shot: String::new(), explanation: String::new(), relation_status: "confirmed".into(),
            evidence_title: String::new(), evidence_kind: "documented".into(), documented: String::new(),
            interpretation: String::new(), bibliography: String::new(), attachment: None,
            card_status: "unreviewed".into(), decision: String::new(), decision_reason: String::new(),
            speaker: String::new(), character: String::new(), speaker_position: "offscreen".into(),
            speaker_note: String::new(), color_label: String::new(), color_scene: String::new(),
            color_phase: String::new(), color_gaze: String::new(), color_judgment: String::new(),
            subject: [0.25, 0.25, 0.25, 0.5], exclusions: Vec::new(), color_result: None,
            query_name: String::new(), query_field: "duration_min".into(), query_value: "0.5".into(),
            query_clauses: Vec::new(), query_match_any: false,
            query_result: None, asr_adapter: "none".into(), asr_status: None, no_dialogue: None,
            asr_segments_json: String::new(), asr_model: String::new(), asr_model_version: String::new(),
            asr_source: String::new(), relation_report: None, evidence_report: None,
            difference_note: String::new() }
    }
}

const KINDS: [(&str, &str); 7] = [("scene", "场景"), ("character", "角色"), ("dialogue", "对白"),
    ("music", "音乐"), ("camera", "运镜"), ("action", "动作阶段"), ("custom", "自定义")];

fn status(value: &str) -> &str {
    match value { "confirmed" => "已核对", "revised" => "已修订", _ => "待核对" }
}

fn color(kind: &str) -> Color32 {
    match kind {
        "scene" => Color32::from_rgb(211, 161, 75), "dialogue" => Color32::from_rgb(96, 173, 213),
        "music" => Color32::from_rgb(165, 128, 214), "action" => Color32::from_rgb(218, 125, 99),
        _ => Color32::from_rgb(109, 180, 149),
    }
}

fn json_text(value: &Value, key: &str) -> String {
    value.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

fn json_u(value: &Value, key: &str) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn json_f(value: &Value, key: &str) -> f64 {
    value.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

fn normalized_rect(rect: Rect, roi: [f32; 4]) -> Rect {
    Rect::from_min_size(
        rect.min + Vec2::new(roi[0].clamp(0.0, 1.0) * rect.width(), roi[1].clamp(0.0, 1.0) * rect.height()),
        Vec2::new(roi[2].clamp(0.0, 1.0) * rect.width(), roi[3].clamp(0.0, 1.0) * rect.height()),
    )
}

fn region_fields(ui: &mut egui::Ui, name: &str, roi: &mut [f32; 4]) {
    ui.horizontal_wrapped(|ui| {
        ui.label(name);
        for (i, label) in ["左 ", "上 ", "宽 ", "高 "].iter().enumerate() {
            let range = if i < 2 { 0.0..=0.99 } else { 0.01..=(1.0 - roi[i - 2]).max(0.01) };
            ui.add(egui::DragValue::new(&mut roi[i]).range(range).speed(0.005).max_decimals(3).prefix(*label));
        }
        roi[2] = roi[2].min(1.0 - roi[0]);
        roi[3] = roi[3].min(1.0 - roi[1]);
    });
}

fn color_sample_geometry_matches(sample: &Value, roi: [f32; 4], subject: [f32; 4], exclusions: &[[f32; 4]]) -> bool {
    // The edit controls are f32. Convert the returned JSON coordinates back to
    // that precision before comparing; their parsed f64 can differ by one ULP.
    <[f32; 4]>::deserialize(&sample["roi"]).ok() == Some(roi)
        && <[f32; 4]>::deserialize(&sample["subject"]).ok() == Some(subject)
        && Vec::<[f32; 4]>::deserialize(&sample["exclusions"]).is_ok_and(|rects| rects == exclusions)
}

fn composition_overlay(painter: &egui::Painter, rect: Rect, subject: &[f32; 4], exclusions: &[[f32; 4]]) {
    for fraction in [1.0 / 3.0, 2.0 / 3.0] {
        let x = rect.left() + rect.width() * fraction;
        let y = rect.top() + rect.height() * fraction;
        painter.vline(x, rect.y_range(), Stroke::new(1.0, Color32::from_white_alpha(110)));
        painter.hline(rect.x_range(), y, Stroke::new(1.0, Color32::from_white_alpha(110)));
    }
    painter.vline(rect.center().x, rect.y_range(), Stroke::new(1.0, Color32::from_white_alpha(150)));
    painter.hline(rect.x_range(), rect.center().y, Stroke::new(1.0, Color32::from_white_alpha(150)));
    painter.rect_stroke(normalized_rect(rect, [0.1, 0.1, 0.8, 0.8]), 0.0,
        Stroke::new(1.0, Color32::from_rgb(255, 207, 97)), egui::StrokeKind::Inside);
    painter.rect_stroke(normalized_rect(rect, *subject), 0.0,
        Stroke::new(2.0, theme::accent()), egui::StrokeKind::Inside);
    for exclusion in exclusions {
        let area = normalized_rect(rect, *exclusion);
        painter.rect_filled(area, 0.0, Color32::from_rgba_unmultiplied(10, 10, 10, 155));
        painter.rect_stroke(area, 0.0, Stroke::new(1.0, theme::warning()), egui::StrokeKind::Inside);
    }
}

fn short(text: &str, count: usize) -> String {
    let mut s: String = text.chars().take(count).collect();
    if text.chars().count() > count { s.push('…'); }
    s.replace('\n', " ")
}

fn x_at(rect: Rect, frame: u64, start: u64, end: u64) -> f32 {
    rect.left() + rect.width() * (frame as f64 - start as f64) as f32 / (end.saturating_sub(start).max(1)) as f32
}

fn frame_at(rect: Rect, x: f32, start: u64, end: u64) -> u64 {
    let t = ((x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0) as f64;
    (start + (t * end.saturating_sub(start) as f64).floor() as u64).min(end.saturating_sub(1))
}

fn view_interval(start: u64, end: u64, frames: u64) -> (u64, u64) {
    let start = start.min(frames.saturating_sub(1));
    (start, end.clamp(start + 1, frames.max(start + 1)))
}

impl Workspace {
    pub fn reset(&mut self) { *self = Self { author: self.author.clone(), ..Self::default() }; }

    pub fn accept(&mut self, method: &str, value: Value, project: &str) -> bool {
        if method.starts_with("study.") && value["project_id"].as_str() != Some(project) { return false; }
        if method == "study.waveform.run" { self.wave = Some(value); return true; }
        if method == "study.exposure.run" {
            if let Some(id) = value["sheet_id"].as_str() { self.sheet_id = id.into(); }
            return true;
        }
        if method == "study.color.run" { self.color_result = Some(value); return true; }
        if method == "study.query.run" { self.query_result = Some(value); return true; }
        if method == "study.no_dialogue" { self.no_dialogue = Some(value); return true; }
        if method == "study.asr_status" { self.asr_status = Some(value); return true; }
        if method == "study.relation.report" { self.relation_report = Some(value); return true; }
        if method == "study.evidence.report" { self.evidence_report = Some(value); return true; }
        if method == "study.evidence.capture" {
            if let (Some(frame), Some(path)) = (value["frame"].as_u64(), value["path"].as_str()) {
                self.attachment = Some((frame, path.into()));
            }
            return false;
        }
        if method == "study.difference.run" {
            let path = value.get("path").and_then(Value::as_str).unwrap_or("");
            self.difference_note = if path.is_empty() { "连续帧导出完成".into() }
                else { format!("已导出清单：{path}") };
            return true;
        }
        false
    }

    pub fn show(&mut self, ui: &mut egui::Ui, state: &State, index: Option<&IndexInfo>, frame: u64,
                texture: Option<&TextureHandle>, thumbs: &HashMap<String, TextureHandle>, stem: &str,
                busy: bool) -> Vec<Action> {
        let mut actions = Vec::new();
        let Some(index) = index else { ui.weak("打开视频后可建立时间标注。"); return actions; };
        let frames = index.frames();
        if frames == 0 { return actions; }
        if self.project != state.project_id {
            self.reset(); self.project = state.project_id.clone(); self.end = frames;
        }
        (self.start, self.end) = view_interval(self.start, self.end, frames);
        if !state.study.tiers.iter().any(|t| t.id == self.tier_id) {
            self.tier_id = state.study.tiers.first().map(|t| t.id.clone()).unwrap_or_default();
        }
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Timeline, "时间层 · 场景");
            ui.selectable_value(&mut self.tab, Tab::Exposure, "动画曝光表");
            ui.selectable_value(&mut self.tab, Tab::Dialogue, "对白 · 音画核对");
            ui.selectable_value(&mut self.tab, Tab::Relations, "镜头关系");
            ui.selectable_value(&mut self.tab, Tab::Evidence, "证据与裁决");
            ui.selectable_value(&mut self.tab, Tab::SoundLinks, "声音链接");
            ui.selectable_value(&mut self.tab, Tab::Color, "色彩 / 构图");
            ui.selectable_value(&mut self.tab, Tab::Query, "跨集 · 查询");
            ui.separator();
            if ui.add_enabled(!busy, egui::Button::new("导出 JSON / EAF / CSV 包…")).clicked() { actions.push(Action::Export); }
            if ui.add_enabled(!busy, egui::Button::new("导回 JSON…")).clicked() { actions.push(Action::ImportJson); }
            if ui.add_enabled(!busy, egui::Button::new("导回 EAF…")).clicked() { actions.push(Action::ImportEaf); }
            ui.label("标注人"); ui.add(egui::TextEdit::singleline(&mut self.author).desired_width(90.0));
        });
        ui.weak("本页使用下方观察范围，不受统计页的镜头筛选影响。源帧从 0 起计、终点不包含；拆并镜头保留标注。");
        egui::ScrollArea::vertical().id_salt("study-body").auto_shrink([false, false]).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("观察范围");
                ui.add(egui::DragValue::new(&mut self.start).range(0..=frames - 1).prefix("起 "));
                ui.add(egui::DragValue::new(&mut self.end).range(1..=frames).prefix("止 "));
                if ui.button("当前镜头").clicked() {
                    if let Some(shot) = state.shots.iter().find(|s| s.start <= frame && frame < s.end) {
                        self.start = shot.start; self.end = shot.end;
                    }
                }
                if ui.button("全片").clicked() { self.start = 0; self.end = frames; }
                if ui.button("此帧为起点").clicked() { self.start = frame; self.end = self.end.max(frame + 1); }
                if ui.button("此帧后为终点").clicked() { self.end = frame + 1; self.start = self.start.min(frame); }
            });
            (self.start, self.end) = view_interval(self.start, self.end, frames);
            self.preview(ui, index, frame, texture, thumbs, stem, state, &mut actions);
            ui.horizontal_wrapped(|ui| {
                if ui.button("◀ 前一帧").clicked() { actions.push(Action::Seek(frame.saturating_sub(1))); }
                if ui.button("后一帧 ▶").clicked() { actions.push(Action::Seek((frame + 1).min(frames - 1))); }
                if ui.button("循环观察范围").clicked() { actions.push(Action::Loop(self.start, self.end)); }
                if ui.button("停止").clicked() { actions.push(Action::Stop); }
                ui.monospace(format!("帧 {frame}  ·  {:.3}s", index.seconds(frame)));
            });
            ui.separator();
            match self.tab {
                Tab::Timeline => {
                    self.timeline(ui, state, frame, &mut actions);
                    self.tiers(ui, state, busy, &mut actions);
                    self.editor(ui, state, frames, busy, &mut actions);
                    self.events(ui, state, false, &mut actions);
                }
                Tab::Exposure => self.exposures(ui, state, frame, busy, &mut actions),
                Tab::Dialogue => {
                    self.dialogue(ui, state, index, frame, busy, &mut actions);
                    self.timeline(ui, state, frame, &mut actions);
                    self.editor(ui, state, frames, busy, &mut actions);
                    self.events(ui, state, true, &mut actions);
                }
                Tab::Relations => self.relations_tab(ui, state, frame, busy, &mut actions),
                Tab::Evidence => self.evidence_tab(ui, state, frame, busy, &mut actions),
                Tab::SoundLinks => self.sound_links_tab(ui, state, busy, &mut actions),
                Tab::Color => self.color_tab(ui, state, frame, busy, &mut actions),
                Tab::Query => self.query_tab(ui, state, busy, &mut actions),
            }
        });
        actions
    }

    fn preview(&mut self, ui: &mut egui::Ui, index: &IndexInfo, frame: u64, texture: Option<&TextureHandle>,
               thumbs: &HashMap<String, TextureHandle>, stem: &str, state: &State, actions: &mut Vec<Action>) {
        ui.horizontal_top(|ui| {
            let (w, h) = index.size();
            let width = ui.available_width().min(420.0);
            let height = (width * h as f32 / w.max(1) as f32).min(220.0);
            let size = Vec2::new(height * w as f32 / h.max(1) as f32, height);
            let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
            ui.painter().rect_filled(rect, 3.0, Color32::BLACK);
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            if let Some(texture) = texture { ui.painter().image(texture.id(), rect, uv, Color32::WHITE); }
            if self.tab == Tab::Exposure {
                if self.onion {
                    for delta in [-2_i64, -1, 1, 2] {
                        let f = frame as i64 + delta;
                        if f < 0 || f >= index.frames() as i64 { continue; }
                        let f = f as u64;
                        if let Some(t) = thumbs.get(&analysis::thumbnail_key(stem, f)) {
                            ui.painter().image(t.id(), rect, uv, Color32::from_white_alpha(45));
                        } else { actions.push(Action::Thumbnail(f)); }
                    }
                }
                if self.roi_pick {
                    if response.drag_started() { self.roi_drag = response.interact_pointer_pos(); }
                    if let (Some(origin), Some(cursor)) = (self.roi_drag, response.interact_pointer_pos()) {
                        let chosen = Rect::from_two_pos(rect.clamp(origin), rect.clamp(cursor));
                        if response.dragged() && chosen.width() >= 2.0 && chosen.height() >= 2.0 {
                            self.roi = [(chosen.left() - rect.left()) / rect.width(), (chosen.top() - rect.top()) / rect.height(),
                                chosen.width() / rect.width(), chosen.height() / rect.height()];
                        }
                    }
                    if response.drag_stopped() { self.roi_drag = None; }
                }
                let [x, y, w, h] = self.roi;
                let roi_rect = Rect::from_min_size(rect.min + Vec2::new(x * rect.width(), y * rect.height()),
                    Vec2::new(w * rect.width(), h * rect.height()));
                ui.painter().rect_stroke(roi_rect, 0.0, Stroke::new(2.0, theme::accent()), egui::StrokeKind::Inside);
            }
            if self.tab == Tab::Color {
                composition_overlay(ui.painter(), rect, &self.subject, &self.exclusions);
                ui.painter().rect_stroke(normalized_rect(rect, self.roi), 0.0,
                    Stroke::new(1.0, Color32::LIGHT_BLUE), egui::StrokeKind::Inside);
            }
            if self.tab == Tab::Relations {
                self.relations_overlay(ui.painter(), rect, state, self.start, self.end);
            }
            ui.vertical(|ui| {
                if self.tab == Tab::Exposure {
                    ui.checkbox(&mut self.roi_pick, "在画面拖拽框选分析区域");
                    ui.checkbox(&mut self.onion, "前后各两帧洋葱皮");
                    if ui.button("重置为全画面").clicked() { self.roi = [0.0, 0.0, 1.0, 1.0]; }
                    ui.small(format!("区域 x {:.2} / y {:.2}\n宽 {:.2} / 高 {:.2}", self.roi[0], self.roi[1], self.roi[2], self.roi[3]));
                    ui.weak("局部变化用于观察口型、身体或背景；不能据此恢复制作时的分层。");
                } else if self.tab == Tab::Color {
                    ui.weak("画面叠加三分线、中心线、安全区、人工主体框与排除区域；自动色值仅作观察。");
                } else {
                    ui.label("点时间条或列表即可回看；画面与声音保持同一源时间。");
                }
            });
        });
    }

    fn relations_overlay(&self, painter: &egui::Painter, rect: Rect, state: &State, start: u64, end: u64) {
        let live: std::collections::HashMap<&str, (u64, u64)> = state.shots.iter()
            .filter(|s| !s.pending).map(|s| (s.id.as_str(), (s.start, s.end))).collect();
        for relation in &state.study.shot_relations {
            let source = json_text(relation, "source_shot_id");
            let target = json_text(relation, "target_shot_id");
            if let (Some(a), Some(b)) = (live.get(source.as_str()), live.get(target.as_str())) {
                let intervals_ok = *a == (json_u(relation, "source_start_frame"), json_u(relation, "source_end_frame"))
                    && *b == (json_u(relation, "target_start_frame"), json_u(relation, "target_end_frame"));
                if intervals_ok && a.1 > start && a.0 < end && b.1 > start && b.0 < end {
                    let from = Pos2::new(x_at(rect, a.0.max(start), start, end), rect.top() + 4.0);
                    let to = Pos2::new(x_at(rect, b.0.max(start), start, end), rect.bottom() - 4.0);
                    painter.line_segment([from, to], Stroke::new(2.0, theme::accent()));
                    painter.circle_filled(to, 3.0, theme::accent());
                }
            }
        }
    }

    fn timeline(&mut self, ui: &mut egui::Ui, state: &State, frame: u64, actions: &mut Vec<Action>) {
        if state.study.tiers.is_empty() { ui.weak("先添加研究模板或自定义层。"); return; }
        ui.label(format!("时间层 · 帧 {}—{}", self.start, self.end));
        for tier in &state.study.tiers {
            let mut events: Vec<&Event> = state.study.events.iter().filter(|e| e.tier_id == tier.id
                && e.start_frame < self.end && self.start < e.end_frame).collect();
            events.sort_by_key(|e| e.start_frame);
            let mut ends: Vec<u64> = Vec::new();
            let mut lanes = Vec::new();
            for e in &events {
                let lane = ends.iter().position(|end| *end <= e.start_frame).unwrap_or(ends.len());
                if lane == ends.len() { ends.push(e.end_frame); } else { ends[lane] = e.end_frame; }
                lanes.push(lane);
            }
            ui.horizontal(|ui| {
                ui.add_sized([75.0, 22.0], egui::Label::new(short(&tier.name, 8)));
                let (rect, response) = ui.allocate_exact_size(Vec2::new(ui.available_width().max(40.0),
                    ends.len().max(1) as f32 * 23.0), Sense::click());
                ui.painter().rect_filled(rect, 2.0, theme::panel());
                let mut hit = false;
                for (e, lane) in events.iter().zip(lanes) {
                    let r = Rect::from_min_max(Pos2::new(x_at(rect, e.start_frame.max(self.start), self.start, self.end), rect.top() + lane as f32 * 23.0),
                        Pos2::new(x_at(rect, e.end_frame.min(self.end), self.start, self.end).max(rect.left() + 1.0), rect.top() + lane as f32 * 23.0 + 21.0));
                    ui.painter().rect_filled(r, 2.0, color(&tier.kind).gamma_multiply(0.6));
                    if r.width() > 35.0 { ui.painter().with_clip_rect(r).text(r.left_center() + Vec2::new(3.0, 0.0), egui::Align2::LEFT_CENTER,
                        short(&e.label, 40), egui::FontId::proportional(12.0), Color32::WHITE); }
                    let item = ui.interact(r, ui.scope_id().with(&e.id), Sense::click());
                    if item.clicked() { self.editing = (*e).clone(); self.tier_id = tier.id.clone(); actions.push(Action::Seek(e.start_frame)); hit = true; }
                    item.on_hover_text(format!("{}\n[{}, {}) · {} · {}", e.label, e.start_frame, e.end_frame, e.speaker, status(&e.review_status)));
                }
                if frame >= self.start && frame < self.end {
                    let x = x_at(rect, frame, self.start, self.end);
                    ui.painter().line_segment([Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())], Stroke::new(1.5, Color32::WHITE));
                }
                if response.clicked() && !hit {
                    if let Some(pos) = response.interact_pointer_pos() { actions.push(Action::Seek(frame_at(rect, pos.x, self.start, self.end))); }
                }
            });
        }
    }

    fn tiers(&mut self, ui: &mut egui::Ui, state: &State, busy: bool, actions: &mut Vec<Action>) {
        ui.collapsing("标注层与词表", |ui| {
            ui.add_enabled_ui(!busy, |ui| {
                if ui.button("添加视听研究模板（场景 / 角色 / 对白 / 音乐 / 运镜 / 动作）").clicked() {
                    actions.push(Action::Rpc("study.template", json!({})));
                }
                ui.horizontal_wrapped(|ui| {
                    ui.label("新层名称"); ui.add(egui::TextEdit::singleline(&mut self.tier_name).desired_width(120.0));
                    egui::ComboBox::from_id_salt("tier-kind").selected_text(KINDS.iter().find(|x| x.0 == self.tier_kind).map(|x| x.1).unwrap_or("自定义")).show_ui(ui, |ui| {
                        for (code, label) in KINDS { ui.selectable_value(&mut self.tier_kind, code.into(), label); }
                    });
                    ui.label("词表（逗号分隔，可空）"); ui.add(egui::TextEdit::singleline(&mut self.vocabulary).desired_width(180.0));
                    if ui.add_enabled(!self.tier_name.trim().is_empty(), egui::Button::new("新建层")).clicked() {
                        let vocabulary: Vec<&str> = self.vocabulary.split([',', '，']).map(str::trim).filter(|s| !s.is_empty()).collect();
                        actions.push(Action::Rpc("study.tier.put", json!({"name": self.tier_name, "kind": self.tier_kind, "vocabulary": vocabulary})));
                        self.tier_name.clear();
                    }
                });
                for tier in &state.study.tiers {
                    ui.horizontal(|ui| {
                        ui.label(format!("{} · {}", tier.name, if tier.vocabulary.is_empty() { "自由标注".into() } else { tier.vocabulary.join(" / ") }));
                        if ui.add_enabled(!state.study.events.iter().any(|e| e.tier_id == tier.id), egui::Button::new("删除空层")).clicked() {
                            actions.push(Action::Rpc("study.tier.remove", json!({"id": tier.id})));
                        }
                    });
                }
            });
        });
    }

    fn editor(&mut self, ui: &mut egui::Ui, state: &State, frames: u64, busy: bool, actions: &mut Vec<Action>) {
        ui.separator();
        ui.add_enabled_ui(!busy && !state.study.tiers.is_empty(), |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("用观察范围新建标注").clicked() {
                    self.editing = Event { start_frame: self.start, end_frame: self.end, review_status: "confirmed".into(), ..Event::default() };
                }
                egui::ComboBox::from_id_salt("event-tier").selected_text(state.study.tiers.iter().find(|t| t.id == self.tier_id).map(|t| t.name.as_str()).unwrap_or("选择层"))
                    .show_ui(ui, |ui| { for tier in &state.study.tiers { ui.selectable_value(&mut self.tier_id, tier.id.clone(), &tier.name); } });
                ui.add(egui::DragValue::new(&mut self.editing.start_frame).range(0..=frames - 1).prefix("起 "));
                ui.add(egui::DragValue::new(&mut self.editing.end_frame).range(1..=frames).prefix("止 "));
                egui::ComboBox::from_id_salt("event-status").selected_text(status(&self.editing.review_status)).show_ui(ui, |ui| {
                    for code in ["unreviewed", "revised", "confirmed"] { ui.selectable_value(&mut self.editing.review_status, code.into(), status(code)); }
                });
            });
            if let Some(tier) = state.study.tiers.iter().find(|t| t.id == self.tier_id).filter(|t| !t.vocabulary.is_empty()) {
                egui::ComboBox::from_id_salt("event-vocab").selected_text(&self.editing.label).show_ui(ui, |ui| {
                    for label in &tier.vocabulary { ui.selectable_value(&mut self.editing.label, label.clone(), label); }
                });
            } else {
                ui.add(egui::TextEdit::multiline(&mut self.editing.label).desired_rows(2).desired_width(f32::INFINITY).hint_text("场景名称、台词、音乐或动作内容"));
            }
            ui.horizontal_wrapped(|ui| {
                ui.label("角色/说话人"); ui.add(egui::TextEdit::singleline(&mut self.editing.speaker).desired_width(110.0));
                ui.label("备注/出处"); ui.add(egui::TextEdit::singleline(&mut self.editing.note).desired_width(260.0));
                let valid = !self.editing.label.trim().is_empty() && self.editing.start_frame < self.editing.end_frame && !self.tier_id.is_empty();
                if ui.add_enabled(valid, egui::Button::new(if self.editing.id.is_empty() { "添加标注" } else { "保存修订" })).clicked() {
                    let mut p = json!({"tier_id": self.tier_id, "start_frame": self.editing.start_frame, "end_frame": self.editing.end_frame,
                        "label": self.editing.label, "speaker": self.editing.speaker, "note": self.editing.note, "author": self.author,
                        "review_status": if self.editing.review_status.is_empty() { "confirmed" } else { &self.editing.review_status }});
                    if !self.editing.id.is_empty() { p["id"] = json!(self.editing.id); }
                    actions.push(Action::Rpc("study.event.put", p));
                    self.editing = Event::default();
                }
                if !self.editing.id.is_empty() && ui.button("删除此标注").clicked() {
                    actions.push(Action::Rpc("study.event.remove", json!({"id": self.editing.id})));
                    self.editing = Event::default();
                }
            });
        });
    }

    fn events(&mut self, ui: &mut egui::Ui, state: &State, dialogue_only: bool, actions: &mut Vec<Action>) {
        ui.horizontal(|ui| { ui.label("搜索内容 / 角色 / 备注"); ui.text_edit_singleline(&mut self.query); });
        let needle = self.query.to_lowercase();
        let mut events: Vec<&Event> = state.study.events.iter().filter(|e| {
            let kind_ok = !dialogue_only || state.study.tiers.iter().any(|t| t.id == e.tier_id && t.kind == "dialogue");
            kind_ok && format!("{} {} {}", e.label, e.speaker, e.note).to_lowercase().contains(&needle)
        }).collect();
        events.sort_by_key(|e| e.start_frame);
        ui.small(format!("{} 条 · 点击编辑，回看不改变标注范围", events.len()));
        egui::ScrollArea::vertical().id_salt("study-events").max_height(200.0).show_rows(ui, 26.0, events.len(), |ui, rows| {
            for row in rows {
                let e = events[row];
                ui.push_id(&e.id, |ui| { ui.horizontal(|ui| {
                    if ui.small_button("回看").clicked() { actions.push(Action::Seek(e.start_frame)); }
                    if ui.small_button("循环").clicked() { actions.push(Action::Loop(e.start_frame, e.end_frame)); }
                    if ui.selectable_label(self.editing.id == e.id, format!("[{}, {}) {} {} · {}", e.start_frame, e.end_frame,
                        e.speaker, short(&e.label, 45), status(&e.review_status))).on_hover_text(format!("{}\n来源：{} · 标注人：{}", e.label, e.source, e.author)).clicked() {
                        self.editing = e.clone(); self.tier_id = e.tier_id.clone();
                    }
                }); });
            }
        });
    }

    fn exposures(&mut self, ui: &mut egui::Ui, state: &State, frame: u64, busy: bool, actions: &mut Vec<Action>) {
        ui.weak("曝光为成片可见变化估计；原画张数无法由此直接推定。每张表最多 5000 帧，自动估计与人工修订分别保留。");
        ui.horizontal_wrapped(|ui| {
            ui.label("名称"); ui.text_edit_singleline(&mut self.sheet_name);
            if ui.add_enabled(!busy && self.end - self.start <= 5000, egui::Button::new("估计所选区域曝光")).clicked() {
                actions.push(Action::Job("study.exposure.run", json!({"start_frame": self.start, "end_frame": self.end,
                    "roi": self.roi, "name": self.sheet_name, "author": self.author}), "曝光估计"));
            }
        });
        if !state.study.exposures.iter().any(|s| s.id == self.sheet_id) {
            self.sheet_id = state.study.exposures.last().map(|s| s.id.clone()).unwrap_or_default();
        }
        egui::ComboBox::from_id_salt("exposure-sheets").selected_text(state.study.exposures.iter().find(|s| s.id == self.sheet_id).map(|s| s.name.as_str()).unwrap_or("尚无曝光表"))
            .show_ui(ui, |ui| { for s in &state.study.exposures { ui.selectable_value(&mut self.sheet_id, s.id.clone(), format!("{} [{}, {})", s.name, s.start_frame, s.end_frame)); } });
        let Some(sheet) = state.study.exposures.iter().find(|s| s.id == self.sheet_id) else { return; };
        ui.label(format!("{} · 自动 {} 段 / 当前 {} 段 · 区域 {:?}", status(&sheet.review_status), sheet.estimated_boundaries.len().saturating_sub(1), sheet.boundaries.len().saturating_sub(1), sheet.roi));
        if let Some(warning) = &sheet.warning { ui.colored_label(theme::warning(), warning); }
        ui.horizontal_wrapped(|ui| {
            if ui.button("查看此表范围与区域").clicked() { self.start = sheet.start_frame; self.end = sheet.end_frame; self.roi = sheet.roi; actions.push(Action::Seek(sheet.start_frame)); }
            if let Some(prev) = sheet.boundaries.iter().rev().find(|&&f| f < frame) {
                if ui.button("上一变化").clicked() { actions.push(Action::Seek(*prev)); }
            }
            if let Some(next) = sheet.boundaries.iter().find(|&&f| f > frame && f < sheet.end_frame) {
                if ui.button("下一变化").clicked() { actions.push(Action::Seek(*next)); }
            }
        });
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal_wrapped(|ui| {
                let internal = sheet.start_frame < frame && frame < sheet.end_frame;
                for (operation, label, enabled) in [("split", "在此帧拆曝光", internal && !sheet.boundaries.contains(&frame)),
                    ("join", "合并此前后曝光", internal && sheet.boundaries.contains(&frame)),
                    ("confirm", "标记已核对", true), ("reset", "恢复自动估计", true), ("remove", "删除曝光表", true)] {
                    if ui.add_enabled(enabled, egui::Button::new(label)).clicked() {
                        actions.push(Action::Rpc("study.exposure.edit", json!({"id": sheet.id, "operation": operation, "frame": frame, "author": self.author})));
                    }
                }
            });
            ui.horizontal_wrapped(|ui| {
                egui::ComboBox::from_id_salt("phase-template")
                    .selected_text(PHASE_TEMPLATES.iter().find(|(code, _)| *code == sheet.phase_template)
                        .map(|(_, label)| *label).unwrap_or("选择动作阶段模板"))
                    .show_ui(ui, |ui| {
                        for (code, label) in PHASE_TEMPLATES {
                            if ui.selectable_label(sheet.phase_template == code, label).clicked() {
                                actions.push(Action::Rpc("study.exposure.edit", json!({
                                    "id": sheet.id, "operation": "template", "template": code, "author": self.author})));
                            }
                        }
                    });
                if ui.button("导出差图 / 连续帧…").clicked() {
                    actions.push(Action::ExportFrames(sheet.id.clone()));
                }
            });
        });
        if !self.difference_note.is_empty() { ui.small(&self.difference_note); }
        if !sheet.phases.is_empty() {
            ui.small(format!("阶段模板 {}：{}", sheet.phase_template, sheet.phases.iter()
                .map(|p| json_text(p, "label")).collect::<Vec<_>>().join(" / ")));
        }
        let (rect, response) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 60.0), Sense::click());
        ui.painter().rect_filled(rect, 2.0, theme::panel());
        for (i, &motion) in sheet.changes.iter().enumerate() {
            let f = sheet.start_frame + i as u64 + 1;
            if f < self.start || f >= self.end { continue; }
            let x = x_at(rect, f, self.start, self.end);
            ui.painter().line_segment([Pos2::new(x, rect.bottom()), Pos2::new(x, rect.bottom() - motion * rect.height())], Stroke::new(1.0, theme::accent()));
        }
        if response.clicked() { if let Some(pos) = response.interact_pointer_pos() { actions.push(Action::Seek(frame_at(rect, pos.x, self.start, self.end))); } }
        egui::ScrollArea::vertical().id_salt("exposure-rows").max_height(220.0).show_rows(ui, 25.0, sheet.boundaries.len().saturating_sub(1), |ui, rows| {
            for i in rows {
                let (a, b) = (sheet.boundaries[i], sheet.boundaries[i + 1]);
                if ui.selectable_label(a <= frame && frame < b, format!("{:04}  ·  [{a}, {b})  ·  保持 {} 帧", i + 1, b - a)).clicked() { actions.push(Action::Seek(a)); }
            }
        });
    }

    fn dialogue(&mut self, ui: &mut egui::Ui, state: &State, index: &IndexInfo, frame: u64, busy: bool, actions: &mut Vec<Action>) {
        ui.horizontal_wrapped(|ui| {
            if ui.add_enabled(!busy, egui::Button::new("加载 SRT…")).clicked() { actions.push(Action::LoadSubtitles); }
            if ui.add_enabled(!busy && !state.subtitle_cues.is_empty(), egui::Button::new("字幕导入对白层")).clicked() {
                actions.push(Action::Rpc("study.import_subtitles", json!({})));
            }
            if ui.add_enabled(!busy, egui::Button::new("载入波形与声画候选")).clicked() {
                actions.push(Action::Job("study.waveform.run", json!({}), "声音观察"));
            }
        });
        let dialogue = union_share(&state.study, "dialogue", index, self.start, self.end);
        let music = union_share(&state.study, "music", index, self.start, self.end);
        ui.label(format!("范围内已标注覆盖：对白 {:.1}% · 音乐 {:.1}%（分别按区间并集计算，可同时存在）", dialogue * 100.0, music * 100.0));
        ui.weak("未标注不代表不存在；导入字幕后可修正时间、台词和说话人，保存修订同步到播放字幕。");
        if let Some(wave) = &self.wave {
            let (rect, response) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 85.0), Sense::click());
            ui.painter().rect_filled(rect, 2.0, theme::panel());
            let start_s = index.seconds(self.start);
            let end_s = index.endpoint_seconds(self.end);
            let x = |s: f64| rect.left() + rect.width() * ((s - start_s) / (end_s - start_s).max(1e-9)) as f32;
            if let Some(points) = wave["waveform"].as_array() {
                for p in points {
                    let t = p["time_s"].as_f64().unwrap_or(0.0);
                    let end = p["end_s"].as_f64().unwrap_or(t);
                    if end < start_s || t > end_s { continue; }
                    let low = p["low"].as_f64().unwrap_or(0.0).clamp(-1.0, 1.0) as f32;
                    let high = p["high"].as_f64().unwrap_or(0.0).clamp(-1.0, 1.0) as f32;
                    let r = Rect::from_min_max(Pos2::new(x(t.max(start_s)), rect.center().y - high * 39.0),
                        Pos2::new(x(end.min(end_s)).max(x(t.max(start_s)) + 1.0), rect.center().y - low * 39.0));
                    ui.painter().rect_filled(r, 0.0, color("dialogue"));
                }
            }
            let cursor = x(index.seconds(frame));
            if rect.x_range().contains(cursor) { ui.painter().vline(cursor, rect.y_range(), Stroke::new(1.0, Color32::WHITE)); }
            if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let t = start_s + ((pos.x - rect.left()) / rect.width()) as f64 * (end_s - start_s);
                    actions.push(Action::Seek(index.frame_at(t)));
                }
            }
            ui.small("混音 PCM 峰值波形（全片最多 6000 栏）；点击定位，循环选区听辨。");
            Self::spectrogram(ui, wave, index, start_s, end_s, actions);
            if let Some(cuts) = wave["cuts"].as_array() {
                egui::ScrollArea::vertical().id_salt("study-cuts").max_height(150.0).show(ui, |ui| {
                    for cut in cuts {
                        let f = cut["frame"].as_u64().unwrap_or(0);
                        if f < self.start || f >= self.end { continue; }
                        let valid = state.shots.windows(2).any(|pair| pair[1].start == f && pair[0].id == cut["left_shot_id"].as_str().unwrap_or("")
                            && pair[1].id == cut["right_shot_id"].as_str().unwrap_or(""));
                        if !valid { continue; }
                        ui.push_id(f, |ui| { ui.horizontal_wrapped(|ui| {
                            if ui.link(format!("帧 {f}")).clicked() { actions.push(Action::Seek(f)); }
                            let candidate = match cut["relation"].as_str().unwrap_or("none") {
                                "J" => "J-cut", "L" => "L-cut", "sync" => "同步", _ => "无明显声学变化",
                            };
                            let offset = cut["sound_offset_s"].as_f64().map(|v| format!("{v:+.2}s")).unwrap_or_else(|| "—".into());
                            ui.label(format!("候选 {candidate} · 偏移 {offset}"));
                            let reviewed = valid_review(&state.study, state, f).unwrap_or("unreviewed");
                            let mut selected = reviewed.to_string();
                            ui.add_enabled_ui(!busy, |ui| {
                                egui::ComboBox::from_id_salt("sound-review").selected_text(relation_label(&selected)).show_ui(ui, |ui| {
                                    for value in ["unreviewed", "J", "L", "sync", "none"] { ui.selectable_value(&mut selected, value.into(), relation_label(value)); }
                                });
                            });
                            if selected != reviewed { actions.push(Action::Rpc("study.sound_review", json!({"frame": f, "relation": selected, "author": self.author}))); }
                        }); });
                    }
                });
            }
        }
    }

    fn spectrogram(ui: &mut egui::Ui, wave: &Value, index: &IndexInfo, start_s: f64, end_s: f64,
                   actions: &mut Vec<Action>) {
        let Some(columns) = wave["spectrogram"]["columns"].as_array() else { return; };
        let (rect, response) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 62.0), Sense::click());
        ui.painter().rect_filled(rect, 2.0, theme::panel());
        let x = |s: f64| rect.left() + rect.width() * ((s - start_s) / (end_s - start_s).max(1e-9)) as f32;
        let bands = wave["spectrogram"]["bands"].as_u64().unwrap_or(1).max(1) as usize;
        for column in columns.iter().filter(|c| {
            let t = c["time_s"].as_f64().unwrap_or(0.0);
            let e = c["end_s"].as_f64().unwrap_or(t);
            e >= start_s && t <= end_s
        }) {
            let t = column["time_s"].as_f64().unwrap_or(0.0);
            let e = column["end_s"].as_f64().unwrap_or(t);
            let left = x(t.max(start_s));
            let right = x(e.min(end_s)).max(left + 1.0);
            let Some(levels) = column["levels"].as_array() else { continue; };
            for (row, level) in levels.iter().enumerate().take(bands) {
                let intensity = level.as_u64().unwrap_or(0) as f32 / 255.0;
                if intensity < 0.05 { continue; }
                let y = rect.bottom() - (row + 1) as f32 * rect.height() / bands as f32;
                ui.painter().rect_filled(Rect::from_min_max(Pos2::new(left, y),
                    Pos2::new(right, y + rect.height() / bands as f32)), 0.0,
                    Color32::from_rgb((70.0 + 170.0 * intensity) as u8,
                        (120.0 + 100.0 * intensity) as u8, (210.0 - 80.0 * intensity) as u8));
            }
        }
        ui.small("波形下方的对数频谱列与来源 PCM 共用源时间轴（NumPy FFT，仅供线索观察）。");
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let t = start_s + ((pos.x - rect.left()) / rect.width()) as f64 * (end_s - start_s);
                actions.push(Action::Seek(index.frame_at(t)));
            }
        }
    }

    fn relations_tab(&mut self, ui: &mut egui::Ui, state: &State, frame: u64, busy: bool, actions: &mut Vec<Action>) {
        ui.weak("镜头关系按人工选择的类型化断言保存；镜头边界变化后旧关系标记为失效，预览只画仍有效的关系。");
        let shots: Vec<&model::ShotRow> = state.shots.iter().filter(|s| !s.pending).collect();
        if self.source_shot.is_empty() { self.source_shot = shots.first().map(|s| s.id.clone()).unwrap_or_default(); }
        if self.target_shot.is_empty() { self.target_shot = shots.get(1).or_else(|| shots.first()).map(|s| s.id.clone()).unwrap_or_default(); }
        ui.add_enabled_ui(!busy && shots.len() >= 2, |ui| {
            ui.horizontal_wrapped(|ui| {
                shot_picker(ui, "起点镜头", &mut self.source_shot, &shots);
                shot_picker(ui, "目标镜头", &mut self.target_shot, &shots);
                combo(ui, "relation-type", "关系类型", &mut self.relation_type, &RELATIONS);
                combo(ui, "transition-type", "转场", &mut self.transition_type,
                    &[("cut", "硬切"), ("dissolve", "叠化"), ("fade_in", "淡入"), ("fade_out", "淡出"),
                      ("wipe", "划像"), ("match_cut", "匹配剪辑"), ("unknown", "待核对")]);
                combo(ui, "relation-status", "核对状态", &mut self.relation_status,
                    &[("confirmed", "已核对"), ("revised", "已修订"), ("unreviewed", "待核对")]);
            });
            ui.horizontal(|ui| {
                ui.label("依据");
                ui.add(egui::TextEdit::singleline(&mut self.explanation).desired_width(f32::INFINITY)
                    .hint_text("视线、动作、构图或转场的依据"));
            });
            if ui.button("保存镜头关系").clicked() {
                actions.push(Action::Rpc("study.relation.put", json!({
                    "relation_type": self.relation_type, "transition_type": self.transition_type,
                    "source_shot_id": self.source_shot, "target_shot_id": self.target_shot,
                    "frame": frame, "explanation": self.explanation, "review_status": self.relation_status,
                    "author": self.author})));
            }
            if ui.button("读取关系与有效性报告").clicked() {
                actions.push(Action::Value("study.relation.report", json!({})));
            }
        });
        if let Some(report) = &self.relation_report {
            ui.small(format!("有效 {} · 失效 {}（失效关系不会被当作当前影片结论）",
                json_u(report, "active"), json_u(report, "stale")));
        }
        for relation in &state.study.shot_relations {
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("#{} → #{} · {} · {}", json_u(relation, "source_shot_number"),
                    json_u(relation, "target_shot_number"), json_text(relation, "relation_type"),
                    status(&json_text(relation, "review_status"))));
                if ui.link(format!("帧 {}", json_u(relation, "frame"))).clicked() {
                    actions.push(Action::Seek(json_u(relation, "frame")));
                }
                ui.small(&json_text(relation, "explanation"));
                if ui.button("删除").clicked() {
                    actions.push(Action::Rpc("study.relation.remove", json!({"id": json_text(relation, "id")})));
                }
            });
        }
    }

    fn evidence_tab(&mut self, ui: &mut egui::Ui, state: &State, frame: u64, busy: bool, actions: &mut Vec<Action>) {
        let card_frame = self.attachment.as_ref().map(|(captured, _)| *captured).unwrap_or(frame);
        ui.weak("证据卡把「记录 / 出处」与「判断 / 解释」分开；裁决保留所有原始意见，不覆盖卡片。");
        ui.add_enabled_ui(!busy, |ui| {
            egui::Grid::new("evidence-editor").num_columns(2).show(ui, |ui| {
                ui.label("证据帧 / 标题");
                ui.horizontal(|ui| {
                    ui.monospace(format!("{card_frame}"));
                    ui.add(egui::TextEdit::singleline(&mut self.evidence_title).desired_width(220.0));
                    combo(ui, "evidence-kind", "类型", &mut self.evidence_kind, &CARD_KINDS);
                    combo(ui, "evidence-status", "核对", &mut self.card_status,
                        &[("unreviewed", "待核对"), ("revised", "已修订"), ("confirmed", "已核对")]);
                });
                ui.end_row();
                ui.label("记录 / 出处");
                ui.add(egui::TextEdit::multiline(&mut self.documented).desired_rows(2).desired_width(f32::INFINITY));
                ui.end_row();
                ui.label("判断 / 解释");
                ui.add(egui::TextEdit::multiline(&mut self.interpretation).desired_rows(2).desired_width(f32::INFINITY));
                ui.end_row();
                ui.label("参考文献 / 附件");
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.bibliography).desired_width(180.0));
                    if ui.button("截取当前帧…").clicked() { actions.push(Action::AttachEvidence(frame)); }
                    if let Some((captured, path)) = &self.attachment {
                        let name = Path::new(path).file_name().unwrap_or_default().to_string_lossy();
                        ui.small(format!("帧 {captured} · {name}"));
                        if ui.small_button("移除附件").clicked() { self.attachment = None; }
                    }
                });
                ui.end_row();
            });
            ui.horizontal_wrapped(|ui| {
                if ui.button("保存证据卡").on_hover_text("有截图时使用截图帧，否则使用当前播放帧").clicked() {
                    actions.push(Action::Rpc("study.evidence.put", json!({
                        "frame": card_frame, "title": self.evidence_title, "kind": self.evidence_kind,
                        "documented": self.documented, "interpretation": self.interpretation,
                        "bibliography": self.bibliography,
                        "attachment_path": self.attachment.as_ref().map(|(_, path)| path),
                        "attachment_frame": self.attachment.as_ref().map(|(frame, _)| frame),
                        "status": self.card_status, "author": self.author})));
                    self.attachment = None;
                }
                if ui.button("读取分歧报告").clicked() {
                    actions.push(Action::Value("study.evidence.report", json!({})));
                }
            });
        });
        if let Some(report) = &self.evidence_report {
            for disagreement in report["disagreements"].as_array().cloned().unwrap_or_default() {
                let f = json_u(&disagreement, "frame");
                ui.horizontal_wrapped(|ui| {
                    if ui.link(format!("帧 {f} 有不同判断")).clicked() { actions.push(Action::Seek(f)); }
                    ui.label(disagreement["reasons"].as_array().map(|rows| rows.iter()
                        .filter_map(Value::as_str).collect::<Vec<_>>().join(" / ")).unwrap_or_default());
                    ui.add(egui::TextEdit::singleline(&mut self.decision).desired_width(160.0).hint_text("裁决结论"));
                    ui.add(egui::TextEdit::singleline(&mut self.decision_reason).desired_width(190.0).hint_text("裁决理由"));
                    if ui.button("记录裁决").clicked() {
                        actions.push(Action::Rpc("study.evidence.adjudicate", json!({
                            "frame": f, "decision": self.decision, "reason": self.decision_reason,
                            "author": self.author, "reviewer": self.author})));
                    }
                });
            }
        }
        for card in &state.study.evidence_cards {
            ui.horizontal_wrapped(|ui| {
                if ui.link(format!("帧 {}", json_u(card, "frame"))).clicked() {
                    actions.push(Action::Seek(json_u(card, "frame")));
                }
                ui.label(format!("{} · {} · {}", json_text(card, "title"), json_text(card, "kind"),
                    status(&json_text(card, "status"))));
                ui.small(&json_text(card, "interpretation"));
                if ui.button("删除").clicked() {
                    actions.push(Action::Rpc("study.evidence.remove", json!({"id": json_text(card, "id")})));
                }
            });
        }
        for adjudication in &state.study.adjudications {
            ui.small(format!("帧 {} 裁决：{}（{}）", json_u(adjudication, "frame"),
                json_text(adjudication, "decision"), json_text(adjudication, "reason")));
        }
    }

    fn sound_links_tab(&mut self, ui: &mut egui::Ui, state: &State, busy: bool, actions: &mut Vec<Action>) {
        ui.weak("声音分层通过人工说话人—角色链接、场景级无对白报告和可选外部 ASR 结果表达；内置粗分不识别身份。");
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("说话人"); ui.add(egui::TextEdit::singleline(&mut self.speaker).desired_width(120.0));
                ui.label("角色"); ui.add(egui::TextEdit::singleline(&mut self.character).desired_width(120.0));
                combo(ui, "speaker-position", "位置", &mut self.speaker_position,
                    &[("onscreen", "画内"), ("offscreen", "画外"), ("unknown", "待核对")]);
                ui.label("出处 / 备注"); ui.add(egui::TextEdit::singleline(&mut self.speaker_note).desired_width(160.0));
                let shot = state.shots.iter().find(|s| s.start <= self.start && self.start < s.end);
                if ui.add_enabled(shot.is_some(), egui::Button::new("保存说话人链接")).clicked() {
                    actions.push(Action::Rpc("study.speaker.put", json!({
                        "start_frame": self.start, "end_frame": self.end,
                        "shot_id": shot.map(|s| s.id.clone()).unwrap_or_default(),
                        "speaker": self.speaker, "character": self.character,
                        "position": self.speaker_position, "note": self.speaker_note,
                        "author": self.author})));
                }
            });
            ui.horizontal_wrapped(|ui| {
                if ui.button("生成无对白报告").clicked() {
                    actions.push(Action::Value("study.no_dialogue", json!({
                        "start_frame": self.start, "end_frame": self.end, "adapter": self.asr_adapter})));
                }
                combo(ui, "asr-adapter", "外部结果来源", &mut self.asr_adapter,
                    &[("none", "不载入"), ("whisperx", "WhisperX 结果"),
                      ("inaSpeechSegmenter", "inaSpeechSegmenter 结果"), ("pyannote", "pyannote 结果")]);
                if ui.button("检查适配器状态").clicked() {
                    actions.push(Action::Value("study.asr_status", json!({"adapter": self.asr_adapter})));
                }
            });
            ui.collapsing("载入用户自备识别片段（不运行模型，不打包权重）", |ui| {
                ui.add(egui::TextEdit::multiline(&mut self.asr_segments_json).desired_rows(2)
                    .desired_width(f32::INFINITY).hint_text("JSON 数组：[{\"start_s\": 1.0, \"end_s\": 2.0, \"text\": \"...\"}]"));
                for (label, field) in [("模型", &mut self.asr_model), ("模型版本", &mut self.asr_model_version),
                                       ("来源", &mut self.asr_source)] {
                    ui.horizontal(|ui| { ui.label(label); ui.text_edit_singleline(field); });
                }
                if ui.button("载入并记录出处").clicked() {
                    let segments: Value = serde_json::from_str(&self.asr_segments_json)
                        .unwrap_or_else(|_| json!({"error": "片段 JSON 格式错误"}));
                    actions.push(Action::Value("study.asr_status", json!({
                        "adapter": self.asr_adapter, "segments": segments,
                        "model": self.asr_model, "model_version": self.asr_model_version,
                        "source": self.asr_source, "author": self.author})));
                }
            });
        });
        if let Some(status) = &self.asr_status {
            ui.small(format!("ASR：{} · {} · {} 条 · {}", json_text(status, "adapter"),
                json_text(status, "status"), status["segments"].as_array().map(|s| s.len()).unwrap_or(0),
                json_text(status, "note")));
        }
        if let Some(report) = &self.no_dialogue {
            let range = &report["range"];
            ui.small(format!("所选范围未标注对白 {} 秒 · {} · {}",
                json_f(range, "no_dialogue_s"), if range["reviewed"].as_bool().unwrap_or(false) { "已核对" } else { "待核对" },
                json_text(range, "note")));
            let scenes = report["scenes"].as_array().cloned().unwrap_or_default();
            if !scenes.is_empty() {
                ui.strong("场景级无对白（scene 层人工标注）");
                for scene in scenes {
                    ui.small(format!("{} · [{}, {}) · 未标注对白 {} 秒 · {}", json_text(&scene, "label"),
                        json_u(&scene, "start_frame"), json_u(&scene, "end_frame"),
                        json_f(&scene, "no_dialogue_s"),
                        if scene["reviewed"].as_bool().unwrap_or(false) { "已核对" } else { "待核对" }));
                }
            }
            for shot in report["shots"].as_array().cloned().unwrap_or_default().iter().take(100) {
                ui.small(format!("#{} · 未标注对白 {} 秒 · {}", json_u(shot, "shot_number"),
                    json_f(shot, "no_dialogue_s"),
                    if shot["reviewed"].as_bool().unwrap_or(false) { "已核对" } else { "待核对" }));
            }
        }
        for link in &state.study.speaker_links {
            ui.horizontal_wrapped(|ui| {
                if ui.link(format!("[{}, {})", json_u(link, "start_frame"), json_u(link, "end_frame"))).clicked() {
                    actions.push(Action::Seek(json_u(link, "start_frame")));
                }
                ui.label(format!("{} → {} · {} · {}", json_text(link, "speaker"), json_text(link, "character"),
                    json_text(link, "position"), json_text(link, "basis")));
                if ui.button("删除").clicked() {
                    actions.push(Action::Rpc("study.speaker.remove", json!({"id": json_text(link, "id")})));
                }
            });
        }
        if let Some(wave) = &self.wave {
            ui.collapsing("声音粗分片段", |ui| {
                for row in wave["segments"].as_array().cloned().unwrap_or_default().iter().take(200) {
                    ui.small(format!("{:.2}–{:.2}s · {} · {}", json_f(row, "start_s"), json_f(row, "end_s"),
                        json_text(row, "label"), json_text(row, "basis")));
                }
            });
        }
    }

    fn color_tab(&mut self, ui: &mut egui::Ui, state: &State, frame: u64, busy: bool, actions: &mut Vec<Action>) {
        ui.weak("用归一化 ROI、黑边/字幕/台标排除掩码和人工主体框计算单帧色值；自动观察与人工判断分别保存。");
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("场景"); ui.add(egui::TextEdit::singleline(&mut self.color_scene).desired_width(110.0));
                ui.label("阶段"); ui.add(egui::TextEdit::singleline(&mut self.color_phase).desired_width(80.0));
                ui.label("标签"); ui.add(egui::TextEdit::singleline(&mut self.color_label).desired_width(110.0));
                ui.label("视线"); ui.add(egui::TextEdit::singleline(&mut self.color_gaze).desired_width(80.0));
            });
            egui::CollapsingHeader::new("分析选区、主体框与排除区").show(ui, |ui| {
                region_fields(ui, "分析选区", &mut self.roi);
                region_fields(ui, "主体框", &mut self.subject);
                let mut remove = None;
                for (i, exclusion) in self.exclusions.iter_mut().enumerate() {
                    ui.push_id(i, |ui| {
                        region_fields(ui, &format!("排除区 {}", i + 1), exclusion);
                        if ui.small_button("移除此排除区").clicked() { remove = Some(i); }
                    });
                }
                if let Some(i) = remove { self.exclusions.remove(i); }
            });
            ui.horizontal_wrapped(|ui| {
                if ui.add_enabled(self.start <= frame && frame < self.end, egui::Button::new("采样当前帧"))
                    .on_disabled_hover_text("请先把播放位置移到观察范围内").clicked() {
                    actions.push(Action::Job("study.color.run", json!({
                        "start_frame": self.start, "end_frame": self.end, "frame": frame,
                        "roi": self.roi, "exclusions": self.exclusions, "subject": self.subject}), "色彩构图采样"));
                }
                if ui.add_enabled(self.exclusions.len() < 12, egui::Button::new("添加排除区（默认底部字幕带）")).clicked() {
                    self.exclusions.push([0.0, 0.85, 1.0, 0.15]);
                }
                if ui.button("清空排除区").clicked() { self.exclusions.clear(); }
                ui.small(format!("主体 {:?} · 排除 {} 个", self.subject, self.exclusions.len()));
            });
            ui.add(egui::TextEdit::multiline(&mut self.color_judgment).desired_rows(2)
                .desired_width(f32::INFINITY).hint_text("人工判断：人物位置、色温、构图如何参与叙事"));
            let sample_matches = self.color_result.as_ref().is_some_and(|sample| {
                sample["frame"].as_u64() == Some(frame) && self.start <= frame && frame < self.end
                    && color_sample_geometry_matches(sample, self.roi, self.subject, &self.exclusions)
            });
            if ui.add_enabled(sample_matches, egui::Button::new("保存观察与判断"))
                .on_disabled_hover_text("请使用当前帧和选区重新采样").clicked() {
                actions.push(Action::Rpc("study.color.put", json!({
                    "start_frame": self.start, "end_frame": self.end, "label": self.color_label,
                    "scene": self.color_scene, "phase": self.color_phase, "gaze": self.color_gaze,
                    "roi": self.roi, "exclusions": self.exclusions, "subject": self.subject,
                    "observation": self.color_result, "judgment": self.color_judgment,
                    "author": self.author})));
            }
        });
        if let Some(result) = &self.color_result {
            ui.small(format!("采样帧 {} · 保留像素 {} · 亮度 {} · 饱和度 {} · 主体区 {}",
                json_u(result, "frame"), json_u(&result["color"], "pixels"),
                result["color"]["luminance"].as_f64().map(|v| format!("{v:.4}")).unwrap_or_else(|| "—".into()),
                result["color"]["saturation"].as_f64().map(|v| format!("{v:.4}")).unwrap_or_else(|| "—".into()),
                json_text(&result["composition"]["subject"], "zone")));
        }
        for row in &state.study.color_observations {
            ui.horizontal_wrapped(|ui| {
                let sampled = row["observation"]["frame"].as_u64();
                let interval = format!("[{}, {})", json_u(row, "start_frame"), json_u(row, "end_frame"));
                let label = sampled.map_or_else(|| format!("{interval} · 旧记录无采样帧"), |f| format!("{interval} · 帧 {f}"));
                if ui.link(label).clicked() {
                    actions.push(Action::Seek(sampled.unwrap_or_else(|| json_u(row, "start_frame"))));
                }
                ui.label(format!("{} · {} · {}", json_text(row, "scene"), json_text(row, "label"),
                    json_text(row, "judgment")));
                ui.small(format!("观察：保留像素 {}", json_u(&row["observation"]["color"], "pixels")));
                if ui.button("删除").clicked() {
                    actions.push(Action::Rpc("study.color.remove", json!({"id": json_text(row, "id")})));
                }
            });
        }
    }

    fn query_tab(&mut self, ui: &mut egui::Ui, state: &State, busy: bool, actions: &mut Vec<Action>) {
        ui.weak("组合查询按当前项目的文字、标签、景别、运镜、时长、静止率、作品信息与时间层类型筛选镜头；跨集比较会复用保存的查询。");
        ui.add_enabled_ui(!busy, |ui| {
            let raw = self.query_value.trim();
            let numeric = self.query_field.starts_with("duration_") || self.query_field.starts_with("stillness_");
            let value = if raw.is_empty() { None } else if numeric {
                raw.parse::<f64>().ok().filter(|n| n.is_finite() && *n >= 0.0
                    && (!self.query_field.starts_with("stillness_") || *n <= 1.0)).map(Value::from)
            } else { Some(json!(raw)) };
            let invalid = !raw.is_empty() && value.is_none();
            let draft = value.map(|value| json!({"field": self.query_field, "value": value}));
            let mut added = false;
            ui.horizontal_wrapped(|ui| {
                ui.label("名称"); ui.add(egui::TextEdit::singleline(&mut self.query_name).desired_width(150.0));
                combo(ui, "query-field", "条件", &mut self.query_field, &QUERY_FIELDS);
                ui.label("值"); ui.add(egui::TextEdit::singleline(&mut self.query_value).desired_width(140.0));
                if ui.add_enabled(draft.is_some() && self.query_clauses.len() < 30, egui::Button::new("添加条件")).clicked() {
                    self.query_clauses.extend(draft.clone());
                    self.query_value.clear();
                    added = true;
                }
            });
            if invalid { ui.colored_label(theme::warning(), "请输入有效数值：时长为非负秒数，静止率为 0 到 1。"); }
            let mut remove = None;
            for (i, clause) in self.query_clauses.iter().enumerate() {
                ui.push_id(i, |ui| ui.horizontal(|ui| {
                    let field = clause["field"].as_str().unwrap_or("");
                    let label = QUERY_FIELDS.iter().find(|(key, _)| *key == field).map_or(field, |(_, label)| *label);
                    ui.label(format!("{label}: {}", clause["value"]));
                    if ui.small_button("移除").clicked() { remove = Some(i); }
                }));
            }
            if let Some(i) = remove { self.query_clauses.remove(i); }
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut self.query_match_any, false, "匹配全部条件");
                ui.selectable_value(&mut self.query_match_any, true, "匹配任一条件");
                let mut clauses = self.query_clauses.clone();
                if !added { clauses.extend(draft); }
                let valid = !invalid && !clauses.is_empty() && clauses.len() <= 30;
                if self.query_match_any && clauses.len() > 1 {
                    clauses = vec![json!({"field": "any", "clauses": clauses})];
                }
                if ui.add_enabled(valid && !self.query_name.trim().is_empty(), egui::Button::new("保存组合查询")).clicked() {
                    actions.push(Action::Rpc("study.query.put", json!({
                        "name": self.query_name, "author": self.author, "clauses": clauses})));
                }
                if ui.add_enabled(valid, egui::Button::new("运行组合查询")).clicked() {
                    actions.push(Action::Value("study.query.run", json!({"clauses": clauses})));
                }
            });
        });
        ui.strong("已保存的查询");
        for query in &state.study.saved_queries {
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("{} · {} 条条件 · {}", json_text(query, "name"),
                    query["clauses"].as_array().map(|c| c.len()).unwrap_or(0), json_text(query, "author")));
                if ui.button("运行").clicked() {
                    actions.push(Action::Value("study.query.run", json!({"id": json_text(query, "id")})));
                }
                if ui.button("删除").clicked() {
                    actions.push(Action::Rpc("study.query.remove", json!({"id": json_text(query, "id")})));
                }
            });
        }
        if let Some(result) = &self.query_result {
            ui.strong(format!("命中 {} / 排除 {} 个镜头", json_u(result, "matched"), json_u(result, "rejected")));
            for shot in result["shots"].as_array().cloned().unwrap_or_default().iter().take(200) {
                ui.horizontal_wrapped(|ui| {
                    if ui.link(format!("#{} [{}, {})", json_u(shot, "number") + 1,
                        json_u(shot, "start"), json_u(shot, "end"))).clicked() {
                        actions.push(Action::Seek(json_u(shot, "start")));
                    }
                    ui.small(format!("{:.3}s · {}", json_f(shot, "duration_s"),
                        shot["tags"].as_array().map(|t| t.iter().filter_map(Value::as_str)
                            .collect::<Vec<_>>().join(" / ")).unwrap_or_default()));
                });
            }
        }
        ui.separator();
        ui.strong("版本对齐与推断单位");
        ui.weak("在研究页的「重复镜头 · 多片比较」选择推断单位、组合查询和人工锚点；构图相似与素材复用分开记录。");
        for alignment in &state.study.version_alignments {
            ui.small(format!("{} ↔ {}：{} 个锚点 · 构图相似 {} · 素材复用 {} · {}",
                json_text(alignment, "first_label"), json_text(alignment, "second_label"),
                alignment["report"]["anchors"].as_array().map(|a| a.len()).unwrap_or(0),
                json_u(&alignment["report"], "composition_similarity"),
                json_u(&alignment["report"], "material_reuse"), json_text(alignment, "version_note")));
        }
        ui.small(format!("项目 provenance：schema v{} · 软件 {} · 源版本 {}",
            state.study.version, json_text(&state.study.provenance, "software_version"),
            json_text(&state.study.provenance, "source_version")));
    }
}

fn shot_picker(ui: &mut egui::Ui, label: &str, selected: &mut String, shots: &[&model::ShotRow]) {
    egui::ComboBox::from_id_salt(label).selected_text(
        shots.iter().find(|s| s.id == *selected).map(|s| format!("#{}", s.number + 1)).unwrap_or_else(|| "选择镜头".into()))
        .show_ui(ui, |ui| {
            for shot in shots {
                ui.selectable_value(selected, shot.id.clone(), format!("#{} [{}, {})", shot.number + 1, shot.start, shot.end));
            }
        });
}

fn combo(ui: &mut egui::Ui, id: &str, label: &str, selected: &mut String, options: &[(&str, &str)]) {
    egui::ComboBox::from_id_salt(id).selected_text(label).show_ui(ui, |ui| {
        for (code, text) in options { ui.selectable_value(selected, (*code).to_string(), *text); }
    });
}

fn relation_label(value: &str) -> &str {
    match value { "J" => "确认 J-cut", "L" => "确认 L-cut", "sync" => "确认同步", "none" => "非声音桥", _ => "待核对" }
}

fn valid_review<'a>(data: &'a Data, state: &State, frame: u64) -> Option<&'a str> {
    let pair = state.shots.windows(2).find(|pair| pair[1].start == frame)?;
    let review = data.sound_reviews.get(&frame.to_string())?;
    if review.signature == vec![json!(pair[0].id), json!(pair[1].id), json!(pair[0].start), json!(frame), json!(pair[1].end)] {
        Some(&review.relation)
    } else { None }
}

fn union_share(data: &Data, kind: &str, index: &IndexInfo, start: u64, end: u64) -> f64 {
    let mut intervals: Vec<(u64, u64)> = data.events.iter().filter(|e| data.tiers.iter().any(|t| t.id == e.tier_id && t.kind == kind)
        && e.start_frame < end && start < e.end_frame).map(|e| (e.start_frame.max(start), e.end_frame.min(end))).collect();
    intervals.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (a, b) in intervals {
        if let Some(last) = merged.last_mut().filter(|last| a <= last.1) { last.1 = last.1.max(b); }
        else { merged.push((a, b)); }
    }
    let seconds: f64 = merged.iter().map(|&(a, b)| index.endpoint_seconds(b) - index.seconds(a)).sum();
    seconds / (index.endpoint_seconds(end) - index.seconds(start)).max(1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn viewport_mapping_uses_half_open_source_frames() {
        let rect = Rect::from_min_size(Pos2::new(100.0, 0.0), Vec2::new(200.0, 40.0));
        assert_eq!(frame_at(rect, 100.0, 50, 150), 50);
        assert_eq!(frame_at(rect, 200.0, 50, 150), 100);
        assert_eq!(frame_at(rect, 300.0, 50, 150), 149);
        assert_eq!(view_interval(80, 20, 100), (80, 81));
    }
    #[test]
    fn new_exposure_result_selects_its_sheet_without_accepting_stale_projects() {
        let mut panel = Workspace::default();
        panel.sheet_id = "first".into();
        assert!(!panel.accept("study.exposure.run", json!({"project_id": "old", "sheet_id": "stale"}), "new"));
        assert_eq!(panel.sheet_id, "first");
        assert!(panel.accept("study.exposure.run", json!({"project_id": "new", "sheet_id": "second"}), "new"));
        assert_eq!(panel.sheet_id, "second");
    }
    #[test]
    fn stale_waveforms_are_not_loaded_into_another_project() {
        let mut panel = Workspace::default();
        assert!(!panel.accept("study.waveform.run", json!({"project_id": "old"}), "new"));
        assert!(panel.wave.is_none());
    }

    #[test]
    fn evidence_attachment_keeps_its_frame_and_ignores_other_projects() {
        let mut panel = Workspace::default();
        panel.accept("study.evidence.capture", json!({"project_id": "p", "frame": 14, "path": "evidence.png"}), "p");
        panel.accept("study.evidence.capture", json!({"project_id": "old", "frame": 19, "path": "wrong.png"}), "p");
        assert_eq!(panel.attachment, Some((14, "evidence.png".into())));
    }

    #[test]
    fn color_sample_matches_round_tripped_geometry_but_not_an_edited_mask() {
        let roi = [0.037, 0.069, 0.714, 0.901];
        let subject = [0.071, 0.127, 0.53, 0.62];
        let exclusions = [[0.0, 0.85, 1.0, 0.15]];
        let sample = json!({"roi": roi, "subject": subject, "exclusions": exclusions});
        let returned = serde_json::from_str(&sample.to_string()).unwrap();
        assert!(color_sample_geometry_matches(&returned, roi, subject, &exclusions));
        assert!(!color_sample_geometry_matches(&returned, roi, subject, &[]));
    }

    #[test]
    fn report_results_use_current_project_and_keep_original_cards() {
        let mut panel = Workspace::default();
        let report = json!({"project_id": "new", "cards": [
            {"id": "a", "interpretation": "视线匹配"}, {"id": "b", "interpretation": "不匹配"}],
            "disagreements": [{"frame": 4, "reasons": ["视线匹配", "不匹配"]}]});
        assert!(!panel.accept("study.evidence.report", json!({"project_id": "old"}), "new"));
        assert!(panel.evidence_report.is_none());
        assert!(panel.accept("study.evidence.report", report.clone(), "new"));
        let stored = panel.evidence_report.as_ref().unwrap();
        assert_eq!(stored["cards"].as_array().unwrap().len(), 2);
        assert_eq!(stored["disagreements"][0]["frame"], 4);
    }
}
