//! 共享边界校正：预览只移动播放器，应用时才提交一个领域编辑。

use eframe::egui;
use crate::model::ShotRow;

#[derive(Clone, Copy, PartialEq)]
pub enum Side { Start, End }

pub struct BoundaryEditor {
    pub project_id: String,
    pub boundary: usize,
    pub original: u64,
    pub restore_frame: u64,
    pub lower: u64,
    pub upper: u64,
    pub side: Side,
    text: String,
    focus: bool,
}

pub enum Action {
    Preview(u64),
    Apply { boundary: usize, frame: u64 },
    Cancel(u64),
}

impl BoundaryEditor {
    pub fn new(project_id: &str, shots: &[ShotRow], row: usize, side: Side, restore_frame: u64) -> Option<Self> {
        let boundary = if side == Side::Start { row } else { row.checked_add(1)? };
        let left = shots.get(boundary.checked_sub(1)?)?;
        let right = shots.get(boundary)?;
        if left.pending || right.pending || left.end != right.start { return None; }
        Some(Self {
            project_id: project_id.to_string(), boundary, original: right.start,
            restore_frame, lower: left.start + 1, upper: right.end - 1,
            side, text: right.start.to_string(), focus: true,
        })
    }

    pub fn valid_for(&self, project_id: &str, shots: &[ShotRow]) -> bool {
        self.project_id == project_id && shots.get(self.boundary)
            .map(|s| !s.pending && s.start == self.original).unwrap_or(false)
            && shots.get(self.boundary - 1).map(|s| !s.pending && s.start + 1 == self.lower).unwrap_or(false)
    }

    pub fn show(&mut self, ctx: &egui::Context) -> Option<Action> {
        let mut open = true;
        let mut result = None;
        let title = if self.side == Side::Start { "校正共享起点" } else { "校正共享终点" };
        egui::Window::new(title).id(egui::Id::unique("boundary-editor"))
            .open(&mut open).collapsible(false).resizable(false).default_width(360.0)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                ui.label("预览相邻源帧；应用后同时调整两侧镜头。");
                ui.weak("终点不包含在当前镜头内，帧号从 0 起。");
                ui.add_space(10.0);
                let parsed = self.text.trim().parse::<u64>().ok();
                ui.horizontal(|ui| {
                    if ui.add_enabled(parsed.map(|f| f > self.lower).unwrap_or(false), egui::Button::new("◁")).clicked() {
                        let frame = parsed.unwrap() - 1;
                        self.text = frame.to_string(); result = Some(Action::Preview(frame));
                    }
                    let input = ui.add(egui::TextEdit::singleline(&mut self.text).desired_width(180.0));
                    if self.focus { input.request_focus(); self.focus = false; }
                    if ui.add_enabled(parsed.map(|f| f < self.upper).unwrap_or(false), egui::Button::new("▷")).clicked() {
                        let frame = parsed.unwrap() + 1;
                        self.text = frame.to_string(); result = Some(Action::Preview(frame));
                    }
                });
                ui.monospace(format!("源帧范围  {} — {}", self.lower, self.upper));
                let frame = self.text.trim().parse::<u64>().ok().filter(|f| self.lower <= *f && *f <= self.upper);
                if frame.is_none() { ui.colored_label(crate::theme::warning(), "输入范围内的整数；相邻镜头各保留至少一帧。"); }
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.add_enabled(frame.is_some(), egui::Button::new("预览")).clicked() {
                        result = frame.map(Action::Preview);
                    }
                    if ui.button("取消").clicked() { result = Some(Action::Cancel(self.restore_frame)); }
                    if ui.add_enabled(frame.is_some(), egui::Button::new("应用").fill(crate::theme::accent_wash())).clicked() {
                        result = frame.map(|frame| Action::Apply { boundary: self.boundary, frame });
                    }
                });
            });
        if !open || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            result = Some(Action::Cancel(self.restore_frame));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shots() -> Vec<ShotRow> {
        [(0, 12), (12, 36), (36, 48)].into_iter().enumerate().map(|(number, (start, end))| {
            ShotRow { number, start, end, ..Default::default() }
        }).collect()
    }

    #[test]
    fn both_sides_share_the_same_boundary_and_keep_one_frame() {
        let shots = shots();
        let start = BoundaryEditor::new("p", &shots, 1, Side::Start, 20).unwrap();
        let end = BoundaryEditor::new("p", &shots, 0, Side::End, 4).unwrap();
        assert_eq!((start.boundary, start.lower, start.upper), (1, 1, 35));
        assert_eq!((end.boundary, end.lower, end.upper), (1, 1, 35));
        assert!(BoundaryEditor::new("p", &shots, 0, Side::Start, 0).is_none());
        assert!(BoundaryEditor::new("p", &shots, 2, Side::End, 0).is_none());
    }

    #[test]
    fn pending_and_changed_projects_cannot_receive_a_preview_edit() {
        let mut shots = shots();
        let draft = BoundaryEditor::new("p", &shots, 1, Side::Start, 20).unwrap();
        assert!(draft.valid_for("p", &shots));
        assert!(!draft.valid_for("new", &shots));
        shots[1].pending = true;
        assert!(BoundaryEditor::new("p", &shots, 0, Side::End, 0).is_none());
        assert!(!draft.valid_for("p", &shots));
    }
}
