//! 视觉主题：移植原版 Theme.qml 的调色板系统（ink/paper/warm/graphite/dusk）
//! 并保留二开版的「影院」琥珀主题，支持运行时切换。

use std::sync::atomic::{AtomicUsize, Ordering};

use eframe::egui;

/// 一套完整配色（对应原版 Theme.qml 的 palette 字段）。
#[derive(Clone, Copy)]
pub struct Palette {
    pub key: &'static str,
    pub title: &'static str,
    pub light: bool,
    pub ink: egui::Color32,
    pub secondary: egui::Color32,
    pub faint: egui::Color32,
    pub line: egui::Color32,
    pub accent: egui::Color32,
    pub selected: egui::Color32,
    pub panel: egui::Color32,
    pub input: egui::Color32,
    pub top: egui::Color32,
    pub matte: egui::Color32,
    pub warning: egui::Color32,
    pub error: egui::Color32,
    pub on_accent: egui::Color32,
}

const fn c(r: u8, g: u8, b: u8) -> egui::Color32 {
    egui::Color32::from_rgb(r, g, b)
}

/// 可选主题列表（顺序即菜单顺序）。
pub const THEME_KEYS: [(&str, &str); 6] = [
    ("cinema", "影院琥珀"),
    ("ink", "深灰"),
    ("paper", "浅色"),
    ("warm", "暖灰"),
    ("graphite", "中性纯色"),
    ("dusk", "横向渐变"),
];

/// 按序号取调色板。
pub fn palette(index: usize) -> Palette {
    match index {
        0 => Palette {
            key: "cinema", title: "影院琥珀", light: false,
            ink: c(232, 230, 225), secondary: c(138, 136, 148), faint: c(104, 102, 112),
            line: c(52, 52, 60), accent: c(232, 163, 61), selected: c(232, 163, 61),
            panel: c(30, 30, 36), input: c(38, 38, 45), top: c(22, 22, 26),
            matte: c(16, 16, 19), warning: c(224, 186, 110), error: c(220, 110, 100),
            on_accent: c(28, 22, 12),
        },
        1 => Palette {
            key: "ink", title: "深灰", light: false,
            ink: c(0xe0, 0xe6, 0xe8), secondary: c(0xa6, 0xb2, 0xb8), faint: c(0x78, 0x89, 0x8f),
            line: c(0x39, 0x45, 0x4b), accent: c(0xa2, 0xc5, 0xc9), selected: c(0xb2, 0xce, 0xd1),
            panel: c(0x24, 0x2e, 0x34), input: c(0x1c, 0x25, 0x2b), top: c(0x20, 0x2a, 0x31),
            matte: c(0x13, 0x1a, 0x1e), warning: c(0xdd, 0xbd, 0x85), error: c(0xef, 0xa9, 0xa3),
            on_accent: c(0x19, 0x29, 0x2e),
        },
        2 => Palette {
            key: "paper", title: "浅色", light: true,
            ink: c(0x28, 0x36, 0x3b), secondary: c(0x58, 0x69, 0x70), faint: c(0x78, 0x87, 0x8a),
            line: c(0xc6, 0xd0, 0xd0), accent: c(0x38, 0x6d, 0x72), selected: c(0x34, 0x66, 0x6b),
            panel: c(0xf0, 0xf4, 0xf2), input: c(0xf8, 0xfa, 0xf8), top: c(0xe9, 0xef, 0xee),
            matte: c(0xd4, 0xdc, 0xda), warning: c(0x88, 0x60, 0x22), error: c(0xa1, 0x46, 0x45),
            on_accent: c(0xff, 0xff, 0xff),
        },
        3 => Palette {
            key: "warm", title: "暖灰", light: false,
            ink: c(0xee, 0xe6, 0xdb), secondary: c(0xc2, 0xb6, 0xa7), faint: c(0x9c, 0x90, 0x82),
            line: c(0x51, 0x4b, 0x43), accent: c(0xd5, 0xb5, 0x8a), selected: c(0xe1, 0xc6, 0x9f),
            panel: c(0x39, 0x35, 0x2f), input: c(0x2c, 0x29, 0x25), top: c(0x39, 0x36, 0x30),
            matte: c(0x21, 0x1f, 0x1c), warning: c(0xe4, 0xbd, 0x79), error: c(0xee, 0xab, 0xa0),
            on_accent: c(0x30, 0x29, 0x1e),
        },
        4 => Palette {
            key: "graphite", title: "中性纯色", light: false,
            ink: c(0xe1, 0xe2, 0xdf), secondary: c(0xad, 0xb1, 0xab), faint: c(0x82, 0x87, 0x7f),
            line: c(0x44, 0x48, 0x42), accent: c(0xbb, 0xca, 0xb1), selected: c(0xbb, 0xca, 0xb1),
            panel: c(0x2c, 0x2f, 0x2b), input: c(0x22, 0x25, 0x21), top: c(0x27, 0x2a, 0x27),
            matte: c(0x18, 0x1b, 0x18), warning: c(0xde, 0xbd, 0x86), error: c(0xea, 0xa6, 0xa2),
            on_accent: c(0x25, 0x30, 0x21),
        },
        _ => Palette {
            key: "dusk", title: "横向渐变", light: false,
            ink: c(0xec, 0xe8, 0xef), secondary: c(0xc0, 0xb6, 0xc8), faint: c(0x98, 0x8e, 0xaa),
            line: c(0x51, 0x47, 0x5e), accent: c(0xe1, 0xbb, 0xa8), selected: c(0xc4, 0xbf, 0xe2),
            panel: c(0x37, 0x30, 0x42), input: c(0x29, 0x25, 0x34), top: c(0x46, 0x37, 0x49),
            matte: c(0x1d, 0x20, 0x29), warning: c(0xe2, 0xbd, 0x8b), error: c(0xf0, 0xa9, 0xb3),
            on_accent: c(0x39, 0x2c, 0x2b),
        },
    }
}

static CURRENT: AtomicUsize = AtomicUsize::new(0);

pub fn current_index() -> usize {
    CURRENT.load(Ordering::Relaxed)
}

pub fn set_current(index: usize) {
    CURRENT.store(index.min(THEME_KEYS.len() - 1), Ordering::Relaxed);
}

pub fn current() -> Palette {
    palette(current_index())
}

// -- 颜色访问器（所有绘制都从这里取，切换主题即时生效） ---------------------

pub fn ink() -> egui::Color32 { current().ink }
pub fn secondary() -> egui::Color32 { current().secondary }
pub fn faint() -> egui::Color32 { current().faint }
pub fn line() -> egui::Color32 { current().line }
pub fn accent() -> egui::Color32 { current().accent }
pub fn selected() -> egui::Color32 { current().selected }
pub fn panel() -> egui::Color32 { current().panel }
pub fn input() -> egui::Color32 { current().input }
pub fn top() -> egui::Color32 { current().top }
pub fn matte() -> egui::Color32 { current().matte }
pub fn warning() -> egui::Color32 { current().warning }
pub fn error() -> egui::Color32 { current().error }
pub fn on_accent() -> egui::Color32 { current().on_accent }

/// 主题键（设置持久化用）。
pub fn current_key() -> &'static str { current().key }

/// 当前主题标题（菜单显示）。
pub fn current_title() -> &'static str { current().title }

/// 强调色的低透明度底（选中行、标签 chip）。
pub fn accent_wash() -> egui::Color32 {
    let accent = accent();
    let panel = panel();
    egui::Color32::from_rgb(
        ((accent.r() as u16 * 45 + panel.r() as u16 * 55) / 100) as u8,
        ((accent.g() as u16 * 45 + panel.g() as u16 * 55) / 100) as u8,
        ((accent.b() as u16 * 45 + panel.b() as u16 * 55) / 100) as u8,
    )
}

/// 悬停底色：主色 7% 混入面板（原版 hover 语义）。
pub fn hover() -> egui::Color32 {
    let accent = accent();
    let base = input();
    egui::Color32::from_rgb(
        ((accent.r() as u16 * 7 + base.r() as u16 * 93) / 100) as u8,
        ((accent.g() as u16 * 7 + base.g() as u16 * 93) / 100) as u8,
        ((accent.b() as u16 * 7 + base.b() as u16 * 93) / 100) as u8,
    )
}

/// 安装主题：视觉、字阶、间距。切换主题时再次调用即可。
pub fn install(ctx: &egui::Context) {
    install_fonts(ctx);
    let p = current();

    let mut visuals = if p.light {
        egui::Visuals::light()
    } else {
        egui::Visuals::dark()
    };
    visuals.panel_fill = p.panel;
    visuals.extreme_bg_color = p.input;
    visuals.faint_bg_color = p.input;
    visuals.window_fill = p.panel;
    visuals.window_stroke = egui::Stroke::new(1.0, p.line);
    visuals.window_shadow = egui::Shadow::NONE;
    visuals.popup_shadow = egui::Shadow::NONE;
    {
        let w = &mut visuals.widgets;
        w.noninteractive.bg_fill = p.top;
        w.noninteractive.fg_stroke = egui::Stroke::new(1.0, p.secondary);
        w.noninteractive.bg_stroke = egui::Stroke::NONE;
        w.inactive.bg_fill = p.input;
        w.inactive.weak_bg_fill = p.input;
        w.inactive.fg_stroke = egui::Stroke::new(1.0, p.ink);
        w.inactive.bg_stroke = egui::Stroke::NONE;
        w.hovered.bg_fill = hover();
        w.hovered.weak_bg_fill = hover();
        w.hovered.fg_stroke = egui::Stroke::new(1.0, p.ink);
        w.hovered.bg_stroke = egui::Stroke::NONE;
        w.active.bg_fill = accent_wash();
        w.active.weak_bg_fill = accent_wash();
        w.active.fg_stroke = egui::Stroke::new(1.0, p.accent);
        w.active.bg_stroke = egui::Stroke::new(1.0, p.accent);
    }
    visuals.selection.bg_fill = accent_wash();
    visuals.selection.stroke = egui::Stroke::new(1.0, p.accent);
    visuals.hyperlink_color = p.accent;
    ctx.set_visuals(visuals);

    ctx.style_mut_of(
        if p.light { egui::Theme::Light } else { egui::Theme::Dark },
        |style| {
            let text = &mut style.text_styles;
            text.insert(egui::TextStyle::Heading, egui::FontId::proportional(16.0));
            text.insert(egui::TextStyle::Body, egui::FontId::proportional(13.0));
            text.insert(egui::TextStyle::Small, egui::FontId::proportional(10.5));
            text.insert(egui::TextStyle::Button, egui::FontId::proportional(13.0));
            text.insert(egui::TextStyle::Monospace, egui::FontId::monospace(12.0));
            style.spacing.item_spacing = egui::vec2(8.0, 6.0);
            style.spacing.button_padding = egui::vec2(10.0, 4.0);
            style.spacing.menu_margin = egui::Margin::same(6);
        },
    );
    ctx.set_theme(if p.light {
        egui::ThemePreference::Light
    } else {
        egui::ThemePreference::Dark
    });
}

/// 注册 CJK 字体（无中文形即豆腐块）。set_fonts 幂等，重复调用安全。
pub fn install_fonts(ctx: &egui::Context) {
    let candidates = [
        "C:/Windows/Fonts/Deng.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palettes_are_distinct_and_switchable() {
        let a = palette(0).accent;
        let b = palette(1).accent;
        assert_ne!(a, b, "不同主题的强调色必须不同");
        set_current(2);
        assert_eq!(current().key, "paper");
        assert!(current().light);
        set_current(0);
        assert_eq!(current().key, "cinema");
    }

    #[test]
    fn wash_and_hover_stay_in_range() {
        for index in 0..THEME_KEYS.len() {
            set_current(index);
            let wash = accent_wash();
            let hover = hover();
            assert!(wash.r() > 0 || wash.g() > 0 || wash.b() > 0);
            assert!(hover.r() > 0 || hover.g() > 0 || hover.b() > 0);
        }
        set_current(0);
    }
}
