//! 界面基础件：图标、图标按钮、主按钮、分段选择、区块标题、标签 chip、卡片。
//! 所有面板都从这里取，按钮的尺寸、圆角、悬停与提示文字才一致。

use eframe::egui::{self, Response, RichText, Ui};

use crate::theme;

/// Phosphor 图标码位（只收界面用到的，名字与 phosphoricons.com 一致）。
pub mod icon {
    pub const ARROW_CLOCKWISE: &str = "\u{E036}";
    pub const ARROW_COUNTER_CLOCKWISE: &str = "\u{E038}";
    pub const ARROWS_IN_LINE_HORIZONTAL: &str = "\u{E530}";
    pub const CAMERA: &str = "\u{E10E}";
    pub const CARET_LEFT: &str = "\u{E138}";
    pub const CARET_RIGHT: &str = "\u{E13A}";
    pub const CHART_BAR: &str = "\u{E150}";
    pub const CHECK: &str = "\u{E182}";
    pub const CHECK_CIRCLE: &str = "\u{E184}";
    pub const CLOSED_CAPTIONING: &str = "\u{E1A4}";
    pub const COLUMNS: &str = "\u{E546}";
    pub const CORNERS_IN: &str = "\u{E1CE}";
    pub const CORNERS_OUT: &str = "\u{E1D0}";
    pub const CROSSHAIR: &str = "\u{E1D6}";
    pub const EXPORT: &str = "\u{EAF0}";
    pub const FILE_CSV: &str = "\u{EB1C}";
    pub const FILE_PDF: &str = "\u{E702}";
    pub const FILM_STRIP: &str = "\u{E792}";
    pub const FLAG: &str = "\u{E244}";
    pub const FLOPPY_DISK: &str = "\u{E248}";
    pub const FOLDER_OPEN: &str = "\u{E256}";
    pub const GIF: &str = "\u{E274}";
    pub const IMAGES: &str = "\u{E836}";
    pub const INFO: &str = "\u{E2CE}";
    pub const KEYBOARD: &str = "\u{E2D8}";
    pub const MONITOR_PLAY: &str = "\u{E58C}";
    pub const NOTE_PENCIL: &str = "\u{E34C}";
    pub const PALETTE: &str = "\u{E6C8}";
    pub const PAUSE: &str = "\u{E39E}";
    pub const PLAY: &str = "\u{E3D0}";
    pub const PLUS: &str = "\u{E3D4}";
    pub const REPEAT: &str = "\u{E3F6}";
    pub const SCISSORS: &str = "\u{EAE0}";
    pub const SIDEBAR: &str = "\u{EAB6}";
    pub const SKIP_BACK: &str = "\u{E5A4}";
    pub const SKIP_FORWARD: &str = "\u{E5A6}";
    pub const SPARKLE: &str = "\u{E6A2}";
    pub const SPEAKER_HIGH: &str = "\u{E44A}";
    pub const SPEAKER_SLASH: &str = "\u{E45A}";
    pub const VIDEO: &str = "\u{E740}";
    pub const WARNING: &str = "\u{E4E0}";
    pub const X: &str = "\u{E4F6}";
}

/// 图标与文字拼成按钮文案。
pub fn label(icon: &str, text: &str) -> String {
    format!("{icon} {text}")
}

/// 悬停提示：启用和禁用时都显示——禁用的按钮最需要说明为什么不能点。
pub trait Tip {
    fn tip(self, text: impl Into<String>) -> Self;
}

impl Tip for Response {
    fn tip(self, text: impl Into<String>) -> Self {
        let text = text.into();
        self.on_hover_text(text.clone()).on_disabled_hover_text(text)
    }
}

/// 方形图标按钮：平时无底，悬停出底。
pub fn icon_button(ui: &mut Ui, icon: &str, enabled: bool, tip: &str) -> Response {
    ui.add_enabled(
        enabled,
        egui::Button::new(RichText::new(icon).size(17.0))
            .frame_when_inactive(false)
            .min_size(egui::vec2(30.0, 28.0)),
    )
    .tip(tip)
}

/// 开关型图标按钮：打开时用强调底和强调色，关闭时同普通图标按钮。
pub fn toggle_icon(ui: &mut Ui, icon: &str, on: bool, enabled: bool, tip: &str) -> Response {
    ui.add_enabled(
        enabled,
        egui::Button::selectable(on, RichText::new(icon).size(17.0))
            .frame_when_inactive(on)
            .min_size(egui::vec2(30.0, 28.0)),
    )
    .tip(tip)
}

/// 图标 + 文字的次要按钮（带底和描边）。
pub fn button(ui: &mut Ui, icon: &str, text: &str, enabled: bool) -> Response {
    ui.add_enabled(enabled, egui::Button::new(label(icon, text)).min_size(egui::vec2(0.0, 28.0)))
}

/// 无底的文字按钮（工具条里次一级的动作）。
pub fn quiet_button(ui: &mut Ui, icon: &str, text: &str, enabled: bool) -> Response {
    ui.add_enabled(
        enabled,
        egui::Button::new(label(icon, text)).frame_when_inactive(false).min_size(egui::vec2(0.0, 28.0)),
    )
}

/// 主按钮：一屏只放一个，标出「下一步该做什么」。
pub fn primary(ui: &mut Ui, icon: &str, text: &str, enabled: bool) -> Response {
    let text = RichText::new(label(icon, text)).color(theme::on_accent());
    ui.add_enabled(
        enabled,
        egui::Button::new(text)
            .fill(theme::accent())
            .stroke(egui::Stroke::NONE)
            .min_size(egui::vec2(0.0, 28.0)),
    )
}

/// 分段选择（互斥的几种视图）。返回值表示本帧是否改了选项。
pub fn segmented<T: PartialEq + Copy, S: AsRef<str>>(ui: &mut Ui, current: &mut T, options: &[(T, S)]) -> bool {
    let mut changed = false;
    egui::Frame::new()
        .fill(theme::input())
        .stroke(egui::Stroke::new(1.0, theme::outline()))
        .corner_radius(7)
        .inner_margin(2)
        .show(ui, |ui| {
            // 放在右到左布局里时子项也会倒着排；倒序遍历，视觉上仍从左到右。
            let reversed = ui.layout().main_dir() == egui::Direction::RightToLeft;
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                let ordered: Vec<&(T, S)> = if reversed { options.iter().rev().collect() } else { options.iter().collect() };
                for (value, text) in ordered {
                    let selected = *current == *value;
                    let response = ui.add(
                        egui::Button::selectable(selected, text.as_ref())
                            .frame_when_inactive(selected)
                            .min_size(egui::vec2(0.0, 24.0)),
                    );
                    if response.clicked() && !selected {
                        *current = *value;
                        changed = true;
                    }
                }
            });
        });
    changed
}

/// 页签栏：选中项强调色文字 + 底部 2px 下划线，未选中为次要色；底部一条分隔线。
pub fn tabs<T: PartialEq + Copy, S: AsRef<str>>(ui: &mut Ui, current: &mut T, options: &[(T, S)]) -> bool {
    let mut changed = false;
    let row = ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 18.0;
        for (value, text) in options {
            let selected = *current == *value;
            let color = if selected { theme::ink() } else { theme::secondary() };
            let font = if selected { theme::bold(13.5) } else { egui::FontId::proportional(13.5) };
            let response = ui.add(
                egui::Label::new(RichText::new(text.as_ref()).font(font).color(color)).sense(egui::Sense::click()),
            );
            let response = response.on_hover_cursor(egui::CursorIcon::PointingHand);
            let underline_y = response.rect.bottom() + 5.0;
            if selected || response.hovered() {
                let stroke = if selected {
                    egui::Stroke::new(2.0, theme::accent())
                } else {
                    egui::Stroke::new(2.0, theme::line())
                };
                ui.painter().hline(response.rect.x_range(), underline_y, stroke);
            }
            if response.clicked() && !selected {
                *current = *value;
                changed = true;
            }
        }
    });
    let y = row.response.rect.bottom() + 6.0;
    ui.painter().hline(ui.max_rect().x_range(), y, egui::Stroke::new(1.0, theme::outline()));
    ui.add_space(10.0);
    changed
}

/// 区块小标题：浅色粗体，与正文拉开层次。
pub fn section(ui: &mut Ui, title: &str) {
    ui.add_space(6.0);
    ui.label(RichText::new(title).font(theme::bold(12.5)).color(theme::secondary()));
    ui.add_space(2.0);
}

/// 小标签（景别、区段、状态）。底色取 `color` 的淡色，文字取 `color`。
pub fn chip(ui: &mut Ui, text: &str, color: egui::Color32) -> Response {
    egui::Frame::new()
        .fill(theme::mix(theme::panel(), color, 0.16))
        .corner_radius(9)
        .inner_margin(egui::Margin::symmetric(7, 1))
        .show(ui, |ui| ui.label(RichText::new(text).small().color(color)))
        .response
}

/// 卡片容器：右栏各区块、跳转框等。
pub fn card<R>(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui) -> R) -> R {
    egui::Frame::new()
        .fill(theme::card())
        .stroke(egui::Stroke::new(1.0, theme::outline()))
        .corner_radius(8)
        .inner_margin(egui::Margin::symmetric(12, 10))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add_contents(ui)
        })
        .inner
}

/// 工具条里的竖分隔。
pub fn divider(ui: &mut Ui) {
    ui.add_space(2.0);
    ui.add(egui::Separator::default().vertical().spacing(6.0));
    ui.add_space(2.0);
}

/// 空状态：图标 + 一句话说明 + 下一步提示，居中。
pub fn empty_state(ui: &mut Ui, icon: &str, title: &str, hint: &str) {
    ui.vertical_centered(|ui| {
        ui.add_space((ui.available_height() * 0.3).max(12.0));
        ui.label(RichText::new(icon).size(34.0).color(theme::faint()));
        ui.add_space(6.0);
        ui.label(RichText::new(title).font(theme::bold(14.0)).color(theme::secondary()));
        ui.label(RichText::new(hint).small().color(theme::faint()));
    });
}
