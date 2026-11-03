//! 视觉主题：「放映室」——暗底 + 胶片琥珀。
//!
//! 依据 app-screen 设计规范：
//! - 色彩只落在数据与主操作上（≤5% 彩色像素），chrome 保持中性；
//! - 表面扁平，用色阶分层，不用线框和阴影；
//! - 无英雄标题；数据诚实（缺失值显示灰色破折号）。

use eframe::egui;

// -- 色板（放映室） -----------------------------------------------------------

/// 地面：放映厅深炭。
pub const GROUND: egui::Color32 = egui::Color32::from_rgb(22, 22, 26);
/// 面板：比地面亮一档的表面。
pub const PANEL: egui::Color32 = egui::Color32::from_rgb(30, 30, 36);
/// 面板抬高一级（弹层、选中行底）。
pub const PANEL_RAISED: egui::Color32 = egui::Color32::from_rgb(38, 38, 45);
/// 正文字（暖白）。
pub const INK: egui::Color32 = egui::Color32::from_rgb(232, 230, 225);
/// 次要文字。
pub const MUTED: egui::Color32 = egui::Color32::from_rgb(138, 136, 148);
/// 胶片琥珀：主色，只用于数据与主操作。
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(232, 163, 61);
/// 琥珀的暗底变体（选中行背景）。
pub const ACCENT_DIM: egui::Color32 = egui::Color32::from_rgb(60, 48, 28);
/// 成功/通过（少量语义色）。
pub const OK: egui::Color32 = egui::Color32::from_rgb(120, 180, 120);
/// 错误。
pub const BAD: egui::Color32 = egui::Color32::from_rgb(220, 110, 100);

/// 安装主题：视觉、字阶、间距一次到位。
pub fn install(ctx: &egui::Context) {
    install_fonts(ctx);

    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = PANEL;
    visuals.extreme_bg_color = GROUND; // 输入框、拖动条槽
    visuals.faint_bg_color = PANEL_RAISED; // 条纹/悬停底
    visuals.window_stroke = egui::Stroke::NONE;
    visuals.window_shadow = egui::Shadow::NONE;
    // 指针状态样式：扁平，色阶表达状态
    {
        let w = &mut visuals.widgets;
        w.noninteractive.bg_fill = GROUND;
        w.noninteractive.fg_stroke = egui::Stroke::new(1.0, MUTED);
        w.noninteractive.bg_stroke = egui::Stroke::NONE;
        w.inactive.bg_fill = PANEL_RAISED;
        w.inactive.weak_bg_fill = PANEL_RAISED;
        w.inactive.fg_stroke = egui::Stroke::new(1.0, INK);
        w.inactive.bg_stroke = egui::Stroke::NONE;
        w.hovered.bg_fill = egui::Color32::from_rgb(46, 46, 54);
        w.hovered.weak_bg_fill = egui::Color32::from_rgb(46, 46, 54);
        w.hovered.fg_stroke = egui::Stroke::new(1.0, INK);
        w.hovered.bg_stroke = egui::Stroke::NONE;
        w.active.bg_fill = ACCENT_DIM;
        w.active.weak_bg_fill = ACCENT_DIM;
        w.active.fg_stroke = egui::Stroke::new(1.0, ACCENT);
        w.active.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    }
    visuals.selection.bg_fill = ACCENT_DIM;
    visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT);
    visuals.hyperlink_color = ACCENT;
    ctx.set_visuals(visuals);

    // 字阶：小标题、紧凑正文（工具型界面，信息密度优先）
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        let text = &mut style.text_styles;
        text.insert(egui::TextStyle::Heading, egui::FontId::proportional(16.0));
        text.insert(egui::TextStyle::Body, egui::FontId::proportional(13.0));
        text.insert(egui::TextStyle::Small, egui::FontId::proportional(10.5));
        text.insert(egui::TextStyle::Button, egui::FontId::proportional(13.0));
        text.insert(egui::TextStyle::Monospace, egui::FontId::monospace(12.0));
        // 间距：紧凑但成组
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(10.0, 4.0);
        style.spacing.menu_margin = egui::Margin::same(6);
    });
}

/// 注册 CJK 字体（无中文形即豆腐块）。set_fonts 幂等，重复调用安全。
pub fn install_fonts(ctx: &egui::Context) {
    let candidates = [
        "C:/Windows/Fonts/Deng.ttf",   // 等线（Win10+ 默认，ttf 稳）
        "C:/Windows/Fonts/simhei.ttf", // 黑体
        "C:/Windows/Fonts/msyh.ttc",   // 微软雅黑
    ];
    let mut bytes = None;
    for path in candidates {
        if let Ok(data) = std::fs::read(path) {
            bytes = Some(data);
            break;
        }
    }
    let Some(bytes) = bytes else {
        eprintln!("警告：未找到系统中文字体，界面中文将显示为方块");
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    fonts
        .font_data
        .insert("system-cjk".into(), egui::FontData::from_owned(bytes).into());
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        if let Some(list) = fonts.families.get_mut(&family) {
            list.push("system-cjk".into());
        }
    }
    ctx.set_fonts(fonts);
}
