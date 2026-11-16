//! 研究工作区：切点审校、人工改动审计、作品信息与区段、重复镜头、多片比较、视觉大模型标注、研究导出。
//! 结果保持为边车返回的 JSON，界面只读需要的字段；所有写操作都以 Action 交回主程序走 RPC。

use std::collections::BTreeMap;
use eframe::egui;
use serde_json::{json, Value};
use crate::model::{self, State};
use crate::theme;

#[derive(Clone, Copy, Default, PartialEq)]
pub enum Tab { #[default] Review, Annotate, Compare, Vlm, Lapian }

#[derive(Default)]
pub struct Research {
    pub tab: Tab,
    pub review: Option<Value>,
    pub audit: Option<Value>,
    pub repeats: Option<Value>,
    pub agreement: Option<Value>,
    pub compare: Option<Value>,
    pub vlm_config: Option<Value>,
    pending_config: Option<Value>,
    pub vlm_report: Option<Value>,
    pub suggestions: Option<Value>,
    pub narrative: Option<Value>,
    narrative_project: String,
    narrative_loaded: bool,
    show_confirmed: bool,
    meta_form: BTreeMap<String, String>,
    meta_project: String,
    compare_group: String,
    compare_include_regions: bool,
    compare_inference_unit: String,
    compare_query_id: String,
    compare_alignment: String,
    compare_alignment_note: String,
    compare_alignment_name: String,
    vlm_form: VlmForm,
    vlm_loaded: bool,
}

#[derive(Default)]
struct VlmForm {
    provider: String,
    model: String,
    base_url: String,
    api_key: String,
    effort: String,
    concurrency: String,
}

pub enum Action {
    Rpc { method: &'static str, params: Value },
    Job { method: &'static str, params: Value, label: &'static str },
    Locate(u64),
    Compare {
        group_by: String,
        include_regions: bool,
        inference_unit: String,
        saved_query_id: String,
        alignment: Vec<Value>,
        alignment_note: String,
    },
    ExportResearch,
    RunVlm { force: bool },
    ExportLapianPackage,
    ImportLapian,
    ExportLapianSheet,
}

/// Parse strictly increasing manual anchors; unparsable lines are ignored.
fn parse_anchors(text: &str) -> Vec<Value> {
    let mut rows = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() < 3 { continue; }
        let (Ok(first), Ok(second)) = (parts[0].parse::<u64>(), parts[1].parse::<u64>()) else { continue };
        let kind = parts[2];
        if first == 0 || second == 0 || !matches!(kind, "composition_similarity" | "material_reuse") { continue; }
        rows.push(json!([first, second, kind, parts[3..].join(", ")]));
    }
    rows
}

pub const META_FIELDS: [(&str, &str); 11] = [
    ("title", "片名"), ("series", "系列"), ("episode", "集数"), ("year", "年份"), ("director", "导演/监督"),
    ("storyboard", "分镜"), ("episode_director", "演出"), ("animation_director", "作画监督"),
    ("studio", "制作公司"), ("source", "版本/来源"), ("notes", "备注"),
];

fn f(value: &Value, key: &str) -> Option<f64> { value.get(key).and_then(Value::as_f64) }
fn u(value: &Value, key: &str) -> u64 { value.get(key).and_then(Value::as_u64).unwrap_or(0) }
fn s<'a>(value: &'a Value, key: &str) -> &'a str { value.get(key).and_then(Value::as_str).unwrap_or("") }

fn interval(value: &Value, key: &str) -> String {
    match value.get(key).and_then(Value::as_array) {
        Some(pair) if pair.len() == 2 => format!("[{:.2}, {:.2}]",
            pair[0].as_f64().unwrap_or(0.0), pair[1].as_f64().unwrap_or(0.0)),
        _ => "—".into(),
    }
}

fn number(value: Option<f64>, digits: usize) -> String {
    value.map(|v| format!("{v:.digits$}")).unwrap_or_else(|| "—".into())
}

impl Research {
    pub fn reset(&mut self) {
        let tab = self.tab;
        *self = Research { tab, ..Default::default() };
    }

    pub fn show(&mut self, ui: &mut egui::Ui, state: &State, busy: bool) -> Vec<Action> {
        let mut actions = Vec::new();
        ui.horizontal(|ui| {
            for (tab, label) in [(Tab::Review, "切点审校"), (Tab::Annotate, "作品与区段"),
                                 (Tab::Compare, "重复镜头 · 多片比较"), (Tab::Vlm, "视觉大模型"),
                                 (Tab::Lapian, "AI 拉片")] {
                ui.selectable_value(&mut self.tab, tab, label);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add_enabled(!busy && state.completed_count > 0, egui::Button::new("导出研究长表…"))
                    .on_hover_text("每镜头一行（カット编号/区段/人工与模型景别/VLM/拍数）+ 方法说明 JSON + 时长列表")
                    .clicked() { actions.push(Action::ExportResearch); }
                if busy { ui.spinner(); }
            });
        });
        ui.separator();
        egui::ScrollArea::vertical().id_salt("research-body").auto_shrink([false, false]).show(ui, |ui| {
            match self.tab {
                Tab::Review => self.review_tab(ui, state, busy, &mut actions),
                Tab::Annotate => self.annotate_tab(ui, state, &mut actions),
                Tab::Compare => self.compare_tab(ui, state, busy, &mut actions),
                Tab::Vlm => self.vlm_tab(ui, state, busy, &mut actions),
                Tab::Lapian => self.lapian_tab(ui, state, &mut actions),
            }
        });
        actions
    }

    fn review_tab(&mut self, ui: &mut egui::Ui, state: &State, busy: bool, actions: &mut Vec<Action>) {
        ui.weak("自动切镜必有误检与漏检。队列按“检测置信度低、相邻镜头极短、切点两侧画面差异小、人工新增、疑似漏检”排序；\
                 逐个核对后点“已核对”，研究导出会记录核对数与人工改动比例。");
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ui.add_enabled(!busy && state.completed_count > 1, egui::Button::new("生成审校队列")).clicked() {
                actions.push(Action::Job { method: "review.run", params: json!({}), label: "审校队列" });
            }
            if ui.add_enabled(!busy, egui::Button::new("人工改动审计")).clicked() {
                actions.push(Action::Rpc { method: "research.audit", params: json!({}) });
            }
            ui.checkbox(&mut self.show_confirmed, "显示已核对");
        });
        if !state.has_detection_record {
            ui.colored_label(theme::warning(), "此项目没有保存检测原始输出（旧项目或手工切分）：只能按镜头长度与画面差异排队，无法计算改动比例。");
        }
        if let Some(audit) = &self.audit {
            ui.add_space(8.0);
            ui.strong("检测 vs 当前切点");
            if audit.get("available").and_then(Value::as_bool) == Some(true) {
                ui.monospace(format!(
                    "检测 {} · 当前 {} · 保留 {} · 删除 {} · 新增 {} · 移动 {} · 人工改动比例 {}",
                    u(audit, "detected_cuts"), u(audit, "current_cuts"), u(audit, "kept"), u(audit, "removed"),
                    u(audit, "added"), u(audit, "moved"),
                    f(audit, "manual_change_ratio").map(|v| format!("{:.1}%", v * 100.0)).unwrap_or("—".into())));
            } else {
                ui.weak("没有检测记录。");
            }
        }
        let Some(review) = &self.review else { return; };
        ui.add_space(8.0);
        ui.strong(format!("已检查 {} 个切点 · 待核 {} · 已核对 {}",
                          u(review, "checked"), u(review, "flagged").saturating_sub(u(review, "confirmed")), u(review, "confirmed")));
        let items = review.get("items").and_then(Value::as_array).cloned().unwrap_or_default();
        egui::Grid::new("review-grid").striped(true).num_columns(5).show(ui, |ui| {
            ui.weak("帧"); ui.weak("类型"); ui.weak("原因"); ui.weak(""); ui.weak(""); ui.end_row();
            for item in items.iter().filter(|i| self.show_confirmed || !i["confirmed"].as_bool().unwrap_or(false)).take(400) {
                let frame = u(item, "frame");
                if ui.link(format!("{frame}")).clicked() { actions.push(Action::Locate(frame)); }
                ui.label(if s(item, "kind") == "missed" { "疑似漏检" } else { "切点" });
                let reasons: Vec<&str> = item["reasons"].as_array().map(|r| r.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
                ui.label(reasons.join("；"));
                let confirmed = item["confirmed"].as_bool().unwrap_or(false);
                if ui.small_button(if confirmed { "撤销核对" } else { "已核对" }).clicked() {
                    actions.push(Action::Rpc { method: "review.confirm", params: json!({"frame": frame, "confirmed": !confirmed}) });
                }
                ui.weak(item.get("score").and_then(Value::as_f64).map(|v| format!("{v:.2}")).unwrap_or_default());
                ui.end_row();
            }
        });
    }

    fn annotate_tab(&mut self, ui: &mut egui::Ui, state: &State, actions: &mut Vec<Action>) {
        if self.meta_project != state.project_id {
            self.meta_project = state.project_id.clone();
            self.meta_form = state.metadata.clone();
        }
        ui.strong("作品信息");
        ui.weak("用于多片比较的分组（如按分镜/演出）与导出的方法说明。");
        egui::Grid::new("meta-grid").num_columns(2).show(ui, |ui| {
            for (key, label) in META_FIELDS {
                ui.label(label);
                ui.add(egui::TextEdit::singleline(self.meta_form.entry(key.to_string()).or_default()).desired_width(320.0));
                ui.end_row();
            }
        });
        if ui.button("保存作品信息").clicked() {
            let values: BTreeMap<_, _> = META_FIELDS.iter().map(|(k, _)| (k.to_string(), self.meta_form.get(*k).cloned().unwrap_or_default())).collect();
            actions.push(Action::Rpc { method: "metadata.set", params: json!({"values": values}) });
        }
        ui.add_space(12.0);
        ui.separator();
        ui.strong("内容区段");
        ui.weak("在镜头列表标记镜头后，用分析页顶部的“标记为区段”添加。勾选“排除区段”后，统计、运动、导出都不计这些镜头。");
        if state.regions.is_empty() { ui.weak("（尚无区段）"); }
        for region in &state.regions {
            ui.horizontal(|ui| {
                ui.label(format!("{} · 帧 {}–{}", model::region_zh(&region.label), region.start_frame, region.end_frame));
                if ui.small_button("定位").clicked() { actions.push(Action::Locate(region.start_frame)); }
                if ui.small_button("删除").clicked() {
                    actions.push(Action::Rpc { method: "regions.remove", params: json!({"id": region.id}) });
                }
            });
        }
        ui.add_space(12.0);
        ui.separator();
        ui.strong("景别一致性（以人工标注为参照）");
        let manual = state.shots.iter().filter(|s| s.scale_manual.is_some()).count();
        ui.weak(format!("已人工标注 {manual} 个镜头。选中镜头后按 1–5（大特写/特写/中景/全景/远景），0 清除。"));
        if ui.add_enabled(manual > 0, egui::Button::new("计算 κ")).clicked() {
            actions.push(Action::Rpc { method: "scale.agreement", params: json!({}) });
        }
        if let Some(agreement) = &self.agreement {
            for (key, label) in [("model_vs_manual", "景别模型 vs 人工"), ("vlm_vs_manual", "VLM vs 人工")] {
                let row = &agreement[key];
                ui.monospace(format!("{label}: n={} · κ={} · 一致率={}", u(row, "n"), number(f(row, "kappa"), 3),
                    f(row, "agreement").map(|v| format!("{:.1}%", v * 100.0)).unwrap_or("—".into())));
            }
            ui.weak(s(agreement, "note"));
        }
    }

    fn compare_tab(&mut self, ui: &mut egui::Ui, state: &State, busy: bool, actions: &mut Vec<Action>) {
        ui.strong("重复镜头（流用 / bank）");
        ui.weak("每个镜头取 25/50/75% 三帧做感知哈希；相邻镜头相似视为误拆，不计入。哈希会存进项目，供跨集找 OP/ED。");
        ui.horizontal(|ui| {
            if ui.add_enabled(!busy && state.completed_count > 0, egui::Button::new("检测重复镜头")).clicked() {
                actions.push(Action::Job { method: "repeats.run", params: json!({}), label: "重复镜头" });
            }
            if ui.add_enabled(!busy, egui::Button::new("读取已存结果")).clicked() {
                actions.push(Action::Rpc { method: "repeats.get", params: json!({}) });
            }
        });
        if let Some(repeats) = &self.repeats {
            let groups = repeats["groups"].as_array().cloned().unwrap_or_default();
            ui.label(format!("{} 组重复（已哈希 {} 个镜头）", groups.len(), u(repeats, "hashed")));
            for group in groups.iter().take(100) {
                ui.horizontal_wrapped(|ui| {
                    ui.weak(format!("×{}", u(group, "count")));
                    for shot in group["shots"].as_array().cloned().unwrap_or_default() {
                        if ui.link(format!("#{}", u(&shot, "shot_number"))).clicked() {
                            actions.push(Action::Locate(u(&shot, "start_frame")));
                        }
                    }
                });
            }
        }
        ui.add_space(12.0);
        ui.separator();
        ui.strong("多片 / 多集比较");
        ui.weak("选择多个 .shotlab 项目：比较中位镜头时长与剪辑率（含区间），两两 Mann-Whitney（Holm 校正）与 Cliff's delta，\
                 按集数排出趋势；已检测重复镜头的项目还会列出共享片段（多为 OP/ED）。");
        ui.horizontal(|ui| {
            ui.label("按字段分组");
            egui::ComboBox::from_id_salt("compare-group").selected_text(
                META_FIELDS.iter().find(|(k, _)| *k == self.compare_group).map(|(_, v)| *v).unwrap_or("不分组"))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.compare_group, String::new(), "不分组");
                    for (key, label) in META_FIELDS { ui.selectable_value(&mut self.compare_group, key.to_string(), label); }
                });
            ui.checkbox(&mut self.compare_include_regions, "包含 OP/ED 等区段");
            if self.compare_inference_unit.is_empty() { self.compare_inference_unit = "film".into(); }
            egui::ComboBox::from_id_salt("inference-unit").selected_text(match self.compare_inference_unit.as_str() {
                "shot" => "镜头", "scene" => "场景", "episode" => "集", _ => "影片",
            }).show_ui(ui, |ui| {
                for (code, label) in [("shot", "镜头"), ("scene", "场景"), ("episode", "集"), ("film", "影片")] {
                    ui.selectable_value(&mut self.compare_inference_unit, code.into(), label);
                }
            });
            ui.label("保存的查询");
            egui::ComboBox::from_id_salt("compare-query").selected_text(
                state.study.saved_queries.iter().find(|q| s(q, "id") == self.compare_query_id)
                    .map(|q| s(q, "name")).unwrap_or("全部镜头")).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.compare_query_id, String::new(), "全部镜头");
                for query in &state.study.saved_queries {
                    ui.selectable_value(&mut self.compare_query_id, s(query, "id").into(), s(query, "name"));
                }
            });
            if ui.add_enabled(!busy, egui::Button::new("选择项目并比较…")).clicked() {
                actions.push(Action::Compare { group_by: self.compare_group.clone(),
                    include_regions: self.compare_include_regions,
                    inference_unit: self.compare_inference_unit.clone(),
                    saved_query_id: self.compare_query_id.clone(),
                    alignment: parse_anchors(&self.compare_alignment),
                    alignment_note: self.compare_alignment_note.clone() });
            }
        });
        ui.horizontal(|ui| {
            ui.label("人工锚点（每行：第一版镜头, 第二版镜头, 类型, 备注）");
            ui.add(egui::TextEdit::multiline(&mut self.compare_alignment).desired_rows(2)
                .desired_width(f32::INFINITY)
                .hint_text("1, 1, composition_similarity, 构图相似\n3, 4, material_reuse, OP 复用"));
        });
        ui.horizontal(|ui| {
            ui.label("版本说明"); ui.add(egui::TextEdit::singleline(&mut self.compare_alignment_note).desired_width(220.0));
            ui.label("对齐名称"); ui.add(egui::TextEdit::singleline(&mut self.compare_alignment_name).desired_width(130.0));
            if ui.add_enabled(!parse_anchors(&self.compare_alignment).is_empty(), egui::Button::new("保存到当前项目")).clicked() {
                actions.push(Action::Rpc { method: "study.alignment.put", params: json!({
                    "name": self.compare_alignment_name, "version_note": self.compare_alignment_note,
                    "first_label": "第一版", "second_label": "第二版",
                    "anchors": parse_anchors(&self.compare_alignment)})});
            }
            ui.weak("当前项目的首条已保存锚点会在未另填时对比较结果生效");
        });
        let Some(result) = &self.compare else { return; };
        ui.small(format!("推断单位：{} · 每片等权：{} · 明确分层 {} 组 · 共享哈希只标为可能素材复用",
            s(result, "inference_unit"), result["equal_weight"]["per_film_equal_weight"].as_bool().unwrap_or(false),
            result["equal_weight"]["strata"].as_array().map(|s| s.len()).unwrap_or(0)));
        if let Some(query) = result.get("query_summary").filter(|q| !q.is_null()) {
            ui.strong(format!("组合查询「{}」逐片命中 {} / {}", s(query, "name"), u(query, "matched"), u(query, "total")));
            for film in query["films"].as_array().cloned().unwrap_or_default() {
                ui.small(format!("{}：{} / {}", s(&film, "label"), u(&film, "matched"), u(&film, "total")));
            }
        }
        if let Some(aligned) = result.get("version_alignment").filter(|a| !a.is_null()) {
            ui.small(format!("版本对齐：构图相似 {} · 素材复用 {} · 插入/删除 {}",
                u(aligned, "composition_similarity"), u(aligned, "material_reuse"),
                u(aligned, "inserted_or_deleted")));
            ui.small(s(aligned, "note"));
        }
        egui::Grid::new("compare-films").striped(true).num_columns(7).show(ui, |ui| {
            for head in ["作品", "镜头", "MSL", "MSL 95%", "IQR", "剪辑率/分", "95%"] { ui.weak(head); }
            ui.end_row();
            for film in result["films"].as_array().cloned().unwrap_or_default() {
                ui.label(s(&film, "label"));
                ui.monospace(format!("{}", u(&film, "shot_count")));
                ui.monospace(number(f(&film, "msl_s"), 2));
                ui.monospace(interval(&film, "msl_ci95_s"));
                ui.monospace(number(f(&film, "iqr_s"), 2));
                ui.monospace(number(f(&film, "cutting_rate_per_min"), 1));
                ui.monospace(interval(&film, "cutting_rate_ci95_per_min"));
                ui.end_row();
            }
        });
        ui.add_space(6.0);
        egui::Grid::new("compare-pairs").striped(true).num_columns(6).show(ui, |ui| {
            for head in ["A", "B", "样本数 A / B", "Cliff's δ", "Mann-Whitney p", "Holm p"] { ui.weak(head); }
            ui.end_row();
            for pair in result["pairs"].as_array().cloned().unwrap_or_default() {
                ui.label(s(&pair, "first"));
                ui.label(s(&pair, "second"));
                ui.label(format!("{} / {}", u(&pair, "first_n"), u(&pair, "second_n"))).on_hover_text(s(&pair, "note"));
                let delta = &pair["cliffs_delta"];
                ui.monospace(format!("{} ({})", number(f(delta, "delta"), 3), s(delta, "magnitude")));
                ui.monospace(number(pair["mann_whitney"].get("p_two_sided").and_then(Value::as_f64), 4));
                ui.monospace(number(pair["mann_whitney"].get("p_holm").and_then(Value::as_f64), 4));
                ui.end_row();
            }
        });
        if let Some(trend) = result["episode_trend"].as_array().filter(|t| !t.is_empty()) {
            ui.add_space(6.0);
            ui.strong("按集数");
            for row in trend {
                ui.monospace(format!("{:<10} MSL {:>6}s · {:>6}/分 · {} 镜头", s(row, "episode"),
                    number(f(row, "msl_s"), 2), number(f(row, "cutting_rate_per_min"), 1), u(row, "shot_count")));
            }
        }
        if let Some(groups) = result["groups"].as_object().filter(|g| !g.is_empty()) {
            ui.add_space(6.0);
            ui.strong("分组汇总");
            for (key, row) in groups {
                ui.monospace(format!("{key}: {} 镜头 · MSL {}s · {}/分", u(row, "shot_count"),
                    number(f(row, "msl_s"), 2), number(f(row, "cutting_rate_per_min"), 1)));
            }
        }
        let shared = result["shared_sequences"].as_array().cloned().unwrap_or_default();
        if !shared.is_empty() {
            ui.add_space(6.0);
            ui.strong("共享镜头序列（possible_material_reuse；可能是 OP/ED/回顾）");
            for run in shared.iter().take(30) {
                ui.monospace(format!("{} #{}–#{} ↔ {} #{}–#{} · {} 镜头",
                    s(run, "first_label"), run["first_shot_numbers"][0], run["first_shot_numbers"][1],
                    s(run, "second_label"), run["second_shot_numbers"][0], run["second_shot_numbers"][1],
                    u(run, "shots")));
                ui.small(format!("{} · {}", s(run, "classification"), s(run, "note")));
            }
        }
        for note in result["notes"].as_array().cloned().unwrap_or_default() {
            ui.small(egui::RichText::new(note.as_str().unwrap_or("")).color(theme::secondary()));
        }
    }

    fn vlm_tab(&mut self, ui: &mut egui::Ui, state: &State, busy: bool, actions: &mut Vec<Action>) {
        ui.weak("像素统计分不清画出来的特写和远景、镜头上摇和赛璐璐拖动、眼动画和标题卡。视觉大模型从每个镜头的几张关键帧给出这些判断，\
                 作为单独的“VLM 推断”层保存，不会覆盖人工标注；可在“作品与区段”里与人工标注算 κ。");
        ui.colored_label(theme::warning(), "运行时会把关键帧图片发送到下面配置的服务商。版权片源请先确认使用条款。");
        if !self.vlm_loaded {
            self.vlm_loaded = true;
            actions.push(Action::Rpc { method: "vlm.config.get", params: json!({}) });
        }
        // 只在新配置到达时回填表单，避免每帧覆盖用户正在输入的内容。
        if let Some(config) = self.pending_config.take() {
            self.vlm_form = VlmForm {
                provider: s(&config, "provider").into(), model: s(&config, "model").into(),
                base_url: s(&config, "base_url").into(), api_key: String::new(),
                effort: s(&config, "effort").into(), concurrency: u(&config, "concurrency").to_string(),
            };
            self.vlm_config = Some(config);
        }
        let has_key = self.vlm_config.as_ref().and_then(|c| c.get("has_key")).and_then(Value::as_bool).unwrap_or(false);
        egui::Grid::new("vlm-config").num_columns(2).show(ui, |ui| {
            ui.label("服务商");
            egui::ComboBox::from_id_salt("vlm-provider").selected_text(match self.vlm_form.provider.as_str() {
                "openai_compatible" => "OpenAI 兼容接口（通义/智谱/本地）", _ => "Anthropic Claude" })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.vlm_form.provider, "anthropic".into(), "Anthropic Claude");
                    ui.selectable_value(&mut self.vlm_form.provider, "openai_compatible".into(), "OpenAI 兼容接口（通义/智谱/本地）");
                });
            ui.end_row();
            ui.label("模型");
            ui.add(egui::TextEdit::singleline(&mut self.vlm_form.model).desired_width(320.0)
                .hint_text(if self.vlm_form.provider == "anthropic" { "claude-opus-5-5" } else { "qwen-vl-max / glm-4v-plus / llava" }));
            ui.end_row();
            if self.vlm_form.provider == "openai_compatible" {
                ui.label("Base URL");
                ui.add(egui::TextEdit::singleline(&mut self.vlm_form.base_url).desired_width(320.0)
                    .hint_text("https://dashscope.aliyuncs.com/compatible-mode/v1"));
                ui.end_row();
            } else {
                ui.label("推理强度");
                egui::ComboBox::from_id_salt("vlm-effort").selected_text(self.vlm_form.effort.clone()).show_ui(ui, |ui| {
                    for level in ["low", "medium", "high"] { ui.selectable_value(&mut self.vlm_form.effort, level.into(), level); }
                });
                ui.end_row();
            }
            ui.label("API Key");
            ui.add(egui::TextEdit::singleline(&mut self.vlm_form.api_key).password(true).desired_width(320.0)
                .hint_text(if has_key { "已保存（留空不修改）" } else { "Claude 可留空改用 ANTHROPIC_API_KEY" }));
            ui.end_row();
            ui.label("并发");
            ui.add(egui::TextEdit::singleline(&mut self.vlm_form.concurrency).desired_width(60.0));
            ui.end_row();
        });
        ui.horizontal(|ui| {
            if ui.button("保存配置").clicked() {
                let mut params = json!({"provider": self.vlm_form.provider, "model": self.vlm_form.model,
                    "base_url": self.vlm_form.base_url, "effort": self.vlm_form.effort,
                    "concurrency": self.vlm_form.concurrency.trim().parse::<u64>().unwrap_or(2)});
                if !self.vlm_form.api_key.trim().is_empty() { params["api_key"] = json!(self.vlm_form.api_key.trim()); }
                self.vlm_form.api_key.clear();
                actions.push(Action::Rpc { method: "vlm.config.set", params });
            }
            if let Some(path) = self.vlm_config.as_ref().map(|c| s(c, "config_path").to_string()) {
                ui.weak(format!("保存在 {path}（不写入项目文件）"));
            }
        });
        ui.add_space(8.0);
        let annotated = state.shots.iter().filter(|s| s.vlm.is_some()).count();
        ui.label(format!("已标注 {annotated} / {} 个已完成镜头（边界改动后对应标注自动失效）", state.completed_count));
        ui.horizontal(|ui| {
            if ui.add_enabled(!busy && state.completed_count > 0, egui::Button::new("标注当前范围")).clicked() {
                actions.push(Action::RunVlm { force: false });
            }
            if ui.add_enabled(!busy && annotated > 0, egui::Button::new("全部重新标注")).clicked() {
                actions.push(Action::RunVlm { force: true });
            }
            if ui.add_enabled(!busy && annotated > 0, egui::Button::new("从标注建议区段")).clicked() {
                actions.push(Action::Rpc { method: "vlm.suggest_regions", params: json!({}) });
            }
        });
        if let Some(report) = &self.vlm_report {
            ui.monospace(format!("本次标注 {} 个 · 失败 {}", u(report, "annotated"),
                report["errors"].as_array().map(|e| e.len()).unwrap_or(0)));
            for error in report["errors"].as_array().cloned().unwrap_or_default().iter().take(10) {
                ui.small(egui::RichText::new(format!("#{} {}", u(error, "shot_number"), s(error, "error"))).color(theme::error()));
            }
        }
        if let Some(suggestions) = &self.suggestions {
            let list = suggestions["suggestions"].as_array().cloned().unwrap_or_default();
            ui.add_space(6.0);
            ui.strong(format!("区段建议（{}）", list.len()));
            for item in list {
                ui.horizontal(|ui| {
                    let label = s(&item, "label").to_string();
                    ui.label(format!("{} · 帧 {}–{} · {} 镜头", model::region_zh(&label),
                        u(&item, "start_frame"), u(&item, "end_frame"), u(&item, "shots")));
                    if ui.small_button("定位").clicked() { actions.push(Action::Locate(u(&item, "start_frame"))); }
                    if ui.small_button("采纳").clicked() {
                        actions.push(Action::Rpc { method: "regions.add", params: json!({"label": label,
                            "start_frame": u(&item, "start_frame"), "end_frame": u(&item, "end_frame")}) });
                    }
                });
            }
        }
        ui.add_space(8.0);
        ui.separator();
        egui::Grid::new("vlm-rows").striped(true).num_columns(6).show(ui, |ui| {
            for head in ["镜头", "景别", "运镜", "内容", "描述", "证据帧"] { ui.weak(head); }
            ui.end_row();
            for shot in state.shots.iter().filter(|s| s.vlm.is_some()).take(500) {
                let vlm = shot.vlm.as_ref().unwrap();
                if ui.link(format!("#{}", shot.number + 1)).clicked() { actions.push(Action::Locate(shot.start)); }
                ui.label(vlm.shot_scale.as_deref().map(model::scale_zh).unwrap_or("—"));
                ui.label(vlm.camera_movement.as_deref().unwrap_or("—"));
                ui.label(vlm.content_type.as_deref().unwrap_or("—"));
                ui.label(&vlm.description);
                let frames = vlm.evidence_frames.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(", ");
                ui.label(if frames.is_empty() { "—".into() } else {
                    format!("{} 帧{}{}", vlm.evidence_count.unwrap_or(vlm.evidence_frames.len() as u64),
                        if vlm.sampling.is_empty() { String::new() } else { format!(" · {}", vlm.sampling) },
                        if frames.is_empty() { String::new() } else { format!(" · [{frames}]") })
                }).on_hover_text("证据帧是源视频显示帧编号；自适应采样长镜头会取更多有界关键帧");
                ui.end_row();
            }
        });
    }

    fn lapian_tab(&mut self, ui: &mut egui::Ui, state: &State, actions: &mut Vec<Action>) {
        ui.weak("把镜头量测（切点/景别/VLM 标注/运动/声音/台词）打包交给多模态大模型做叙事级拉片：\
                 场景划分、剧情线、观众情绪曲线与全片结构。结果存为独立叙事层，可整体撤销，不覆盖镜头笔记。");
        if self.narrative_project != state.project_id { self.narrative_loaded = false; }
        if !self.narrative_loaded {
            self.narrative_loaded = true;
            self.narrative_project = state.project_id.clone();
            actions.push(Action::Rpc { method: "narrative.get", params: json!({}) });
        }
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ui.button("刷新").clicked() {
                actions.push(Action::Rpc { method: "narrative.get", params: json!({}) });
            }
            if ui.add_enabled(state.completed_count > 0, egui::Button::new("导出分析包…"))
                .on_hover_text("prompt.md + schema.json + manifest 镜头证据表 + 画面拼图 + 字幕；交给任意多模态大模型")
                .clicked() { actions.push(Action::ExportLapianPackage); }
            if ui.add_enabled(state.completed_count > 0, egui::Button::new("导入 AI 结果…"))
                .on_hover_text("大模型返回的 JSON 写回叙事层（dry run 请用 CLI/MCP）")
                .clicked() { actions.push(Action::ImportLapian); }
            if ui.add_enabled(true, egui::Button::new("导出拉片长图…"))
                .on_hover_text("情绪曲线 + 剧情线泳道 + 场景表 PNG")
                .clicked() { actions.push(Action::ExportLapianSheet); }
        });
        let Some(narrative) = &self.narrative else {
            ui.add_space(8.0);
            ui.weak("（叙事层为空）从「导出分析包」开始：把包交给 ChatGPT / GLM / Kimi 等，\
                     用包内 prompt.md 附带的开场白提问，收到的 JSON 用「导入 AI 结果」写回。");
            return;
        };
        let scenes = narrative["scenes"].as_array().cloned().unwrap_or_default();
        if scenes.is_empty() {
            ui.add_space(8.0);
            ui.weak("叙事层还没有场景。");
            return;
        }
        let identity = &narrative["identity"];
        ui.add_space(6.0);
        ui.strong(format!("{} · {} 场景 · {} 曲线点{}",
            s(identity, "title"), scenes.len(),
            narrative["curve_points"].as_array().map(|p| p.len()).unwrap_or(0),
            if s(narrative, "model").is_empty() { String::new() } else { format!(" · {}", s(narrative, "model")) }));
        let lines: Vec<String> = narrative["story_lines"].as_array().cloned().unwrap_or_default()
            .iter().map(|l| s(l, "title").to_string()).collect();
        if !lines.is_empty() { ui.small(format!("剧情线：{}", lines.join(" / "))); }
        ui.add_space(6.0);
        egui::Grid::new("lapian-scenes").striped(true).num_columns(6).show(ui, |ui| {
            for head in ["类型", "镜头", "时间", "标题", "作用", ""] { ui.weak(head); }
            ui.end_row();
            for scene in &scenes {
                ui.label(s(scene, "type"));
                ui.monospace(format!("S{}–S{}", u(scene, "start_shot"), u(scene, "end_shot")));
                ui.monospace(format!("{}s–{}s", number(scene.get("start_s").and_then(Value::as_f64), 1),
                    number(scene.get("end_s").and_then(Value::as_f64), 1)));
                ui.label(s(scene, "title")).on_hover_text(format!("{}\n{}", s(scene, "key_beats"), s(scene, "techniques")));
                ui.small(s(scene, "function"));
                let start_shot = u(scene, "start_shot") as usize;
                if let Some(shot) = state.shots.get(start_shot.saturating_sub(1)) {
                    if ui.small_button("定位").clicked() { actions.push(Action::Locate(shot.start)); }
                }
                ui.end_row();
            }
        });
        ui.add_space(10.0);
        ui.separator();
        ui.strong("观众情绪曲线");
        egui::Grid::new("lapian-curve").striped(true).num_columns(5).show(ui, |ui| {
            for head in ["时间", "强度", "效价", "节奏/情绪", "说明"] { ui.weak(head); }
            ui.end_row();
            for point in narrative["curve_points"].as_array().cloned().unwrap_or_default() {
                ui.monospace(number(point.get("time").and_then(Value::as_f64), 1));
                ui.monospace(number(point.get("intensity").and_then(Value::as_f64), 0));
                ui.monospace(number(point.get("valence").and_then(Value::as_f64), 0));
                ui.label(format!("{} / {}", s(&point, "rhythm_role"), s(&point, "emotion_type")));
                ui.small(format!("{} · {}", s(&point, "title"), s(&point, "why")));
                ui.end_row();
            }
        });
        ui.add_space(10.0);
        ui.separator();
        ui.strong("全片结构");
        for (key, label) in [("overall_structure", "结构"), ("narrative_strategy", "叙事策略"),
                             ("rhythm_pattern", "节奏"), ("information_strategy", "信息策略"),
                             ("core_intent", "核心意图")] {
            let text = s(&narrative["macro"], key);
            if !text.is_empty() { ui.horizontal_wrapped(|ui| { ui.weak(format!("{label}:")); ui.label(text); }); }
        }
    }

    /// 边车结果按方法名落到对应面板；返回 true 表示需要刷新项目状态。
    pub fn accept(&mut self, method: &str, value: Value) -> bool {
        match method {
            "review.run" => { self.review = Some(value); false }
            "review.confirm" => { if let Some(review) = value.get("review").filter(|v| !v.is_null()) { self.review = Some(review.clone()); } true }
            "research.audit" => { self.audit = Some(value); false }
            "repeats.run" | "repeats.get" => { self.repeats = Some(value); method == "repeats.run" }
            "scale.agreement" => { self.agreement = Some(value); false }
            "compare.run" => { self.compare = Some(value); false }
            "vlm.config.get" | "vlm.config.set" => { self.pending_config = Some(value); false }
            "vlm.run" => { self.vlm_report = Some(value); true }
            "vlm.suggest_regions" => { self.suggestions = Some(value); false }
            "narrative.get" => { self.narrative = Some(value); false }
            "lapian.import" | "lapian.deep_import" => { self.narrative_loaded = false; true }
            "lapian.package" | "export.lapian_sheet" => false,
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_route_to_their_panels_and_flag_state_refresh() {
        let mut research = Research::default();
        assert!(!research.accept("review.run", json!({"items": [], "checked": 3})));
        assert_eq!(u(research.review.as_ref().unwrap(), "checked"), 3);
        // 确认后边车返回新的队列快照，并且需要刷新项目状态（dirty）。
        assert!(research.accept("review.confirm", json!({"confirmed": 1, "review": {"checked": 4}})));
        assert_eq!(u(research.review.as_ref().unwrap(), "checked"), 4);
        assert!(research.accept("repeats.run", json!({"groups": []})));
        assert!(!research.accept("repeats.get", json!({"groups": []})));
        assert!(research.accept("vlm.run", json!({"annotated": 2})));
        assert!(!research.accept("vlm.config.get", json!({"provider": "anthropic"})));
        assert!(research.pending_config.is_some());
    }

    #[test]
    fn lapian_results_update_narrative_and_refetch_after_import() {
        let mut research = Research::default();
        research.narrative_loaded = true;
        assert!(!research.accept("narrative.get", json!({"scenes": [{"title": "色块对峙"}]})));
        assert_eq!(research.narrative.as_ref().unwrap()["scenes"].as_array().unwrap().len(), 1);
        // 导入写回项目状态（true → 刷新），并让面板下次重新拉取叙事层。
        assert!(research.accept("lapian.import", json!({"imported": true})));
        assert!(!research.narrative_loaded);
        assert!(!research.accept("lapian.package", json!({"out_dir": "x"})));
    }

    #[test]
    fn reset_keeps_the_tab_but_drops_results() {
        let mut research = Research { tab: Tab::Vlm, ..Default::default() };
        research.accept("compare.run", json!({"films": []}));
        research.reset();
        assert!(research.compare.is_none());
        assert!(research.tab == Tab::Vlm);
    }

    #[test]
    fn interval_formatting_handles_missing_values() {
        assert_eq!(interval(&json!({"x": [1.0, 2.5]}), "x"), "[1.00, 2.50]");
        assert_eq!(interval(&json!({"x": null}), "x"), "—");
        assert_eq!(number(None, 2), "—");
    }

    #[test]
    fn manual_alignment_parser_rejects_invalid_rows_and_keeps_kinds() {
        let rows = parse_anchors("1, 1, composition_similarity, 构图\n3, 4, material_reuse, OP 复用\nx, 5, other, 无效\n0, 0, material_reuse, 无效");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1][2], "material_reuse");
        assert_eq!(rows[1][3], "OP 复用");
    }
}
