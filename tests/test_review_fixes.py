"""2026-11-03 审查修复的回归测试：运动缓存失效、GIF 时间坐标、PDF 分页、中文字体、脚本编码。"""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from shotlab.fonts import CANDIDATES, ui_font
from shotlab.rpc import METHODS
from tests.test_motion_slit import session_with_shots

ROOT = Path(__file__).resolve().parents[1]


class ReviewFixTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.session = session_with_shots(self.directory.name)

    def test_motion_result_is_stale_after_undo_and_redo(self):
        session = self.session
        session.motion_run(session.analysis_scope({}))
        self.assertEqual(len(METHODS["motion.get"](session, {})["shots"]), 3)
        METHODS["edit.undo"](session, {})
        with self.assertRaises(ValueError):
            METHODS["motion.get"](session, {})
        session.motion_run(session.analysis_scope({}))
        METHODS["edit.redo"](session, {})
        with self.assertRaises(ValueError):
            METHODS["export.motion_spectrum"](session, {"path": str(Path(self.directory.name) / "m.png")})

    def test_motion_key_tracks_boundaries_not_just_scope(self):
        session = self.session
        session.motion_run(session.analysis_scope({}))
        # 绕过 _edit 的缓存清理，只验证键本身能识别边界变化。
        session.project.move_boundary(1, 10)
        with self.assertRaises(ValueError):
            session.motion_current({})

    def test_gif_range_uses_origin_relative_seconds(self):
        session = self.session
        shifted = [pts + 500 for pts in session.index.pts]
        session.index = replace(session.index, pts=shifted)
        result = METHODS["export.gif_range"](session, {"start_frame": 0, "end_frame": 12})
        self.assertEqual(result["start_s"], 0.0)
        tb = session.index.tb
        expected = float((shifted[11] - shifted[0] + session.index.durations[11]) * tb)
        self.assertAlmostEqual(result["duration_s"], expected)
        later = METHODS["export.gif_range"](session, {"start_frame": 12, "end_frame": 24})
        self.assertAlmostEqual(later["start_s"], float((shifted[12] - shifted[0]) * tb))

    def test_pdf_sheet_fills_every_column_of_a_page(self):
        session = self.session
        for frame in (3, 6, 9, 15, 18, 21, 24, 27, 30, 33, 39):
            session._edit("拆分", mutate=lambda frame=frame: session.project.split(frame))
        self.assertEqual(session.project.completed_count, 14)
        # 96x54 夹具 → 240 宽缩略图高 135，单元格 165；600 高的页放 3 行 × 4 列 = 12 个镜头。
        result = METHODS["export.pdf_sheet"](session, {
            "path": str(Path(self.directory.name) / "sheet.pdf"), "columns": 4, "page_height": 600})
        self.assertEqual((result["written"], result["pages"]), (14, 2))


class FontTests(unittest.TestCase):
    def test_label_font_has_cjk_glyphs(self):
        if not any(Path(path).exists() for path in CANDIDATES[:-1]):
            self.skipTest("本机没有 CJK 字体")
        font = ui_font(16)
        missing = bytes(font.getmask("\U0010FFFD"))
        self.assertNotEqual(bytes(font.getmask("镜")), missing)
        self.assertNotEqual(bytes(font.getmask("镜")), bytes(font.getmask("头")))


class ScriptEncodingTests(unittest.TestCase):
    def test_powershell_scripts_carry_utf8_bom(self):
        # Windows PowerShell 5.1 按 ANSI 读无 BOM 脚本，中文注释会吞掉下一行。
        for script in sorted((ROOT / "gui").glob("*.ps1")):
            with self.subTest(script=script.name):
                self.assertEqual(script.read_bytes()[:3], b"\xef\xbb\xbf")


if __name__ == "__main__":
    unittest.main()
