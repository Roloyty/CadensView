"""AI 拉片协议回归：分析包导出、容错导入、叙事层撤销与持久化、深拆、长图。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from shotlab import lapian
from shotlab.rpc import METHODS, Session
from tests.test_motion_slit import session_with_shots


def model_payload(session, **overrides):
    payload = {
        "filmIdentity": lapian.build_identity(session),
        "story_lines": [{"id": "protagonist_action", "title": "方块行动线", "description": "测试线"}],
        "scenes": [
            {"start_shot": 1, "end_shot": 2, "title": "色块对峙", "type": "开场",
             "function": "建立红蓝对抗", "primary_line": "protagonist_action",
             "shared_lines": ["protagonist_action"], "techniques": "S2 中点切镜制造节拍",
             "key_beats": "00:00.5 红蓝交替", "confidence": 0.7},
            {"start_shot": 3, "end_shot": 3, "title": "蓝调收束", "type": "结尾",
             "function": "以蓝场收尾", "primary_line": "protagonist_action",
             "techniques": "长镜延续", "key_beats": "00:01.5 定格", "confidence": 0.6},
        ],
        "curve_points": [
            {"time": 0, "intensity": 30, "valence": -10, "emotion_type": "curiosity",
             "rhythm_role": "setup", "title": "开场好奇", "why": "新画面"},
            {"time": "0:01.5", "intensity": 70, "valence": 20, "emotion_type": "tension",
             "rhythm_role": "peak", "title": "切换峰值", "why": "切镜加速"},
        ],
        "macro": {"overall_structure": "红蓝三段式", "writing_lessons": ["对比剪辑"]},
    }
    payload.update(overrides)
    return payload


def fenced(payload):
    return "以下是分析结果：\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


class LapianTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.session = session_with_shots(self.directory.name)
        self.addCleanup(self.session.close_current_store)
        self.rows = self.session.state()["shots"]
        self.assertEqual(len(self.rows), 3)

    def call(self, method, **params):
        return METHODS[method](self.session, params)

    # -- 导出 ------------------------------------------------------------------

    def test_package_export_writes_protocol_files(self):
        out = Path(self.directory.name) / "pkg-frames"
        result = lapian.export_package(self.session, {"out_dir": str(out), "mode": "frames"})
        self.assertTrue((out / "prompt.md").exists())
        schema = json.loads((out / "schema.json").read_text(encoding="utf-8"))
        self.assertIn("filmIdentity", schema)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["identity"]["shot_count"], 3)
        self.assertEqual(len(manifest["shots"]), 3)
        first = manifest["shots"][0]
        self.assertEqual(first["number"], 1)
        self.assertAlmostEqual(first["start_s"], 0.0, places=3)
        self.assertAlmostEqual(first["end_s"], 0.5, places=3)
        self.assertTrue(first["image"].startswith("frames/shot-0001_"))
        self.assertTrue((out / first["image"]).exists())
        self.assertFalse((out / "subtitles.srt").exists())  # 没加载字幕就没有该文件
        prompt = (out / "prompt.md").read_text(encoding="utf-8")
        self.assertIn("事实纪律", prompt)
        self.assertIn("filmIdentity", prompt)
        self.assertEqual(result["shots"], 3)

    def test_package_export_sheets_and_sampling(self):
        out = Path(self.directory.name) / "pkg-sheets"
        result = lapian.export_package(self.session, {"out_dir": str(out), "mode": "sheets",
                                                      "max_images": 2})
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["evidence"]["image_mode"], "sheets")
        self.assertIn("sampling", manifest["evidence"])
        sampled = [s for s in manifest["shots"] if s.get("image")]
        self.assertEqual(len(sampled), 2)
        self.assertTrue(list((out / "sheets").glob("sheet-*.jpg")))
        self.assertEqual(result["image_files"], 1)  # 2 格只够一张拼图

    def test_manifest_carries_subtitles_and_vlm(self):
        srt = Path(self.directory.name) / "sub.srt"
        srt.write_text("1\n00:00:00,500 --> 00:00:01,200\n红色阶段\n", encoding="utf-8")
        METHODS["subtitles.load"](self.session, {"path": str(srt)})
        self.session.vlm_labels = {self.rows[0]["id"]: dict(
            start=self.rows[0]["start"], end=self.rows[0]["end"], shot_scale="FS",
            camera_movement="static", description="纯色场", confidence=0.9)}
        manifest = lapian.build_manifest(self.session, {})
        by_number = {s["number"]: s for s in manifest["shots"]}
        # 字幕 0.5-1.2s 落在镜头 2（0.5s 起）
        self.assertEqual(by_number[2]["dialogue"], "红色阶段")
        self.assertIsNone(by_number[3]["dialogue"])
        self.assertEqual(by_number[1]["shot_scale"], "FS")
        self.assertEqual(manifest["evidence"]["subtitles"], True)

    # -- 容错导入 ----------------------------------------------------------------

    def test_import_roundtrip_preview_undo_and_persistence(self):
        text = fenced({"result": model_payload(self.session)})  # 围栏 + result 包装双层
        preview = self.call("lapian.import", text=text, dry_run=True)
        self.assertTrue(preview["dry_run"])
        self.assertEqual(preview["preview"]["scenes"], 2)
        self.assertEqual(preview["preview"]["curve_points"], 2)
        self.assertEqual(self.session.narrative, {})  # 预览不写回

        result = self.call("lapian.import", text=text, model="测试模型")
        self.assertTrue(result["imported"])
        narrative = self.call("narrative.get")
        self.assertEqual(len(narrative["scenes"]), 2)
        self.assertEqual(narrative["scenes"][0]["title"], "色块对峙")
        self.assertAlmostEqual(narrative["scenes"][0]["start_s"], 0.0, places=3)
        self.assertEqual(narrative["model"], "测试模型")
        peak = [p for p in narrative["curve_points"] if p["rhythm_role"] == "peak"]
        self.assertAlmostEqual(peak[0]["time"], 1.5, places=3)  # '0:01.5' 时间码已解析
        self.assertEqual(narrative["macro"]["overall_structure"], "红蓝三段式")
        # 自定义剧情线被采纳，且未提供的默认线不会重复出现
        self.assertEqual([line["id"] for line in narrative["story_lines"]],
                         ["protagonist_action"])

        # 撤销链：一次导入一条记录
        METHODS["edit.undo"](self.session, {})
        self.assertEqual(self.call("narrative.get")["scenes"], [])

        # 持久化：layer 随项目保存/重开
        self.call("lapian.import", text=text)
        target = Path(self.directory.name) / "saved.shotlab"
        self.session.save_as(str(target))
        reopened = Session()
        self.addCleanup(reopened.close_current_store)
        reopened.open_project(str(target))
        self.assertEqual(len(lapian.data(reopened)["scenes"]), 2)

    def test_import_write_notes_replaces_not_duplicates(self):
        text = json.dumps(model_payload(self.session), ensure_ascii=False)
        self.call("lapian.import", text=text, write_notes=True)
        note = self.session.project.shots[0].note
        self.assertIn("色块对峙", note)
        self.call("lapian.import", text=text, write_notes=True)
        self.assertEqual(self.session.project.shots[0].note.count("色块对峙"), 1)

    def test_import_rejects_wrong_film_and_count(self):
        other = model_payload(self.session)
        other["filmIdentity"]["source"] = "别的片子.mkv"
        with self.assertRaisesRegex(ValueError, "影片不匹配"):
            self.call("lapian.import", text=json.dumps(other, ensure_ascii=False))
        wrong_count = model_payload(self.session)
        wrong_count["filmIdentity"]["shot_count"] = 99
        with self.assertRaisesRegex(ValueError, "镜头数不匹配"):
            self.call("lapian.import", text=json.dumps(wrong_count, ensure_ascii=False))

    def test_time_anchoring_clamps_and_drops(self):
        payload = model_payload(self.session, scenes=[
            {"start_time": "0:00.6", "end_time": "0:01.2", "title": "时间锚定场", "type": "冲突"},
            {"start_shot": 2, "end_shot": 9, "title": "越界场", "type": "过渡"},
            {"title": "无定位场", "type": "过渡"},
        ], curve_points=[])
        _narrative, _warnings, preview = lapian.normalize_analysis(self.session, payload)
        scenes = _narrative["scenes"]
        self.assertEqual(len(scenes), 2)  # 无定位场被丢弃
        self.assertEqual(scenes[0]["anchored_by"], "time")
        self.assertEqual((scenes[0]["start_shot"], scenes[0]["end_shot"]), (2, 2))
        self.assertEqual((scenes[1]["start_shot"], scenes[1]["end_shot"]), (2, 3))  # 9 收敛到 3
        self.assertTrue(preview["warnings"])

    def test_curve_normalization_counts_unknowns(self):
        payload = model_payload(self.session, curve_points=[
            {"time": 0.2, "intensity": 250, "valence": -500, "emotion_type": "rage",
             "rhythm_role": "boom", "title": "越界点"},
            {"time": "无效", "title": "坏点"},
        ])
        narrative, _warnings, preview = lapian.normalize_analysis(self.session, payload)
        self.assertEqual(len(narrative["curve_points"]), 1)
        self.assertEqual(narrative["curve_points"][0]["intensity"], 100)
        self.assertEqual(narrative["curve_points"][0]["valence"], -100)
        self.assertEqual(preview["unknown_emotion_types"], 1)
        self.assertEqual(preview["unknown_rhythm_roles"], 1)

    def test_extract_json_and_unwrap(self):
        self.assertEqual(lapian.extract_json('噪声 {"a": {"b": 1}} 尾噪声'), {"a": {"b": 1}})
        self.assertEqual(lapian.extract_json('前文 {"a": "含{花括号}"} 后'), {"a": "含{花括号}"})
        self.assertEqual(lapian.unwrap_payload({"data": {"scenes": []}}), {"scenes": []})
        self.assertEqual(lapian.unwrap_payload({"macro": {}}), {"macro": {}})

    # -- 深拆与长图（P2/P3） -----------------------------------------------------

    def test_deep_export_and_import(self):
        self.call("lapian.import", text=json.dumps(model_payload(self.session), ensure_ascii=False))
        out = Path(self.directory.name) / "deep"
        result = self.call("lapian.deep", out_dir=str(out), scene_index=0)
        self.assertTrue((out / "prompt.md").exists())
        self.assertTrue((out / "scene.json").exists())
        self.assertEqual(len(list((out / "frames").glob("shot-*.jpg"))), 2)  # 场景含镜头 1-2
        self.assertEqual(result["scene"]["start_shot"], 1)

        reply = {"scene_id": result["scene"]["id"], "title": "色块对峙·精修",
                 "shot_cards": [{"shot": 1, "action": "纯红场", "dialogue": "", "technique": "静帧"},
                                {"shot": 7, "action": "越界卡"}]}
        imported = self.call("lapian.deep_import", text=json.dumps(reply, ensure_ascii=False))
        self.assertEqual(imported["shot_cards"], 1)  # 越界卡被丢弃并警告
        scene = self.call("narrative.get")["scenes"][0]
        self.assertEqual(scene["title"], "色块对峙·精修")
        self.assertEqual(scene["shot_cards"][0]["shot"], 1)

    def test_narrative_compare_across_projects(self):
        # 当前项目导入叙事层并保存；再在独立目录建一个无叙事层的对照项目。
        self.call("lapian.import", text=json.dumps(model_payload(self.session), ensure_ascii=False))
        mine = Path(self.directory.name) / "mine.shotlab"
        self.session.save_as(str(mine))
        other_dir = Path(self.directory.name) / "other-proj"
        other_dir.mkdir()
        other = session_with_shots(str(other_dir))
        other_path = other_dir / "other.shotlab"
        other.save_as(str(other_path))
        other.close_current_store()

        self.session.close_current_store()
        reopened = Session()
        self.addCleanup(reopened.close_current_store)
        reopened.open_project(str(mine))
        result = METHODS["narrative.compare"](reopened, {"others": [str(other_path)]})
        films = {film["label"]: film for film in result["films"]}
        self.assertEqual(len(films), 2)
        with_narrative = next(f for f in result["films"] if f["has_narrative"])
        self.assertEqual(with_narrative["scenes"], 2)
        self.assertEqual(with_narrative["peaks"], 1)
        self.assertEqual(with_narrative["type_distribution"], {"开场": 1, "结尾": 1})
        without = next(f for f in result["films"] if not f["has_narrative"])
        self.assertEqual(without["curve_points"], 0)

    def test_draw_sheet_png(self):
        with self.assertRaisesRegex(ValueError, "叙事层为空"):
            self.call("export.lapian_sheet", path=str(Path(self.directory.name) / "x.png"))
        self.call("lapian.import", text=json.dumps(model_payload(self.session), ensure_ascii=False))
        target = Path(self.directory.name) / "拉片长图.png"
        result = self.call("export.lapian_sheet", path=str(target))
        self.assertTrue(target.stat().st_size > 2000)
        from PIL import Image
        with Image.open(target) as image:
            image.verify()
        self.assertEqual(result["scenes"], 2)

    def test_run_direct_requires_configuration(self):
        with patch("shotlab.vlm.load_config", return_value={}):
            with self.assertRaisesRegex(ValueError, "直连需要"):
                lapian.run_direct(self.session, {})

    def test_fuzz_locks_non_dict_payload_and_string_identity(self):
        """随机 payload 模糊发现的两个崩溃类：顶层非对象、filmIdentity 是字符串。"""
        with self.assertRaisesRegex(ValueError, "顶层不是对象"):
            lapian.normalize_analysis(self.session, 2.5)
        with self.assertRaisesRegex(ValueError, "顶层不是对象"):
            lapian.normalize_analysis(self.session, ["列表也不行"])
        payload = model_payload(self.session, filmIdentity="就是一段字符串")
        narrative, _warnings, _preview = lapian.normalize_analysis(self.session, payload)
        json.dumps(narrative, allow_nan=False)  # 无身份信息按缺失处理而非崩溃
        payload = model_payload(self.session, scenes=2.5)  # scenes 容器本身是标量
        narrative, _warnings, preview = lapian.normalize_analysis(self.session, payload)
        self.assertEqual(narrative["scenes"], [])
        self.assertIn("scenes 跳过 1", "\n".join(preview["warnings"]))

    def test_rpc_serve_roundtrip_sync_and_job(self):
        """GUI 走 serve() 行协议：同步 lapian.import 与 job 型 lapian.package 都要通。"""
        import io

        from shotlab.rpc import serve

        def request(payload):
            sink = io.StringIO()
            serve(io.StringIO(json.dumps(payload) + "\n"), sink, self.session)
            return [json.loads(line) for line in sink.getvalue().splitlines() if line.strip()]

        reply = json.dumps(model_payload(self.session), ensure_ascii=False)
        answer = request({"id": 7, "method": "lapian.import",
                          "params": {"text": reply, "model": "协议测试"}})
        self.assertTrue(answer[0]["ok"], answer[0])
        self.assertEqual(answer[0]["result"]["preview"]["scenes"], 2)
        # 同步第三个位置参数是 emit（函数对象），不能被误当 cancel 用
        self.assertTrue(answer[0]["result"]["imported"])

        from tests.test_motion_slit import run_job
        out = str(Path(self.directory.name) / "serve-pkg")
        done = run_job(self.session, "lapian.package", {"out_dir": out, "mode": "sheets"})
        self.assertEqual(done["phase"], "done", done)
        self.assertEqual(done["result"]["shots"], 3)
        self.assertTrue((Path(out) / "manifest.json").exists())

    # -- 时间码工具 ---------------------------------------------------------------

    def test_time_helpers(self):
        self.assertEqual(lapian.parse_time("52:30"), 3150.0)
        self.assertEqual(lapian.parse_time("1:02:03.5"), 3723.5)
        self.assertIsNone(lapian.parse_time("坏值"))
        self.assertIsNone(lapian.parse_time(-5))
        self.assertEqual(lapian.timecode(75.4), "01:15.4")
        self.assertEqual(lapian.file_timecode(75.4), "01m15s4")

    # -- 对抗性回归：撤销完整性 / 垃圾输入 / 边界 ----------------------------------

    def test_undo_after_write_notes_restores_notes_and_narrative(self):
        """write_notes 会改镜头笔记，撤销必须把笔记和叙事层一起退回去。

        回归：原先走 Session._annotate，它在 mutate 之后才快照 project（前后同值），
        笔记改动撤销不回去。
        """
        text = json.dumps(model_payload(self.session), ensure_ascii=False)
        self.call("lapian.import", text=text, write_notes=True)
        self.assertIn("色块对峙", self.session.project.shots[0].note)
        undo_depth = len(self.session.history.undo)
        self.assertGreaterEqual(undo_depth, 1)
        METHODS["edit.undo"](self.session, {})
        self.assertEqual(self.session.project.shots[0].note, "")
        self.assertEqual(self.call("narrative.get")["scenes"], [])

    def test_reimport_identical_content_adds_no_extra_undo_record(self):
        text = json.dumps(model_payload(self.session), ensure_ascii=False)
        self.call("lapian.import", text=text)
        depth = len(self.session.history.undo)
        self.call("lapian.import", text=text)
        self.assertEqual(len(self.session.history.undo), depth)  # 幂等：无新增撤销记录

    def test_import_survives_junk_items_in_arrays(self):
        """模型偶发在数组里塞字符串/数字，导入应跳过并警告而不是 AttributeError。"""
        payload = model_payload(self.session,
                                story_lines=["垃圾", {"id": "protagonist_action", "title": "线"}],
                                scenes=["垃圾场景", {"start_shot": 1, "end_shot": 3, "title": "唯一场",
                                                     "type": "开场"}, 42],
                                curve_points=[None, {"time": 0.5, "intensity": 50}])
        narrative, _warnings, preview = lapian.normalize_analysis(self.session, payload)
        self.assertEqual(len(narrative["scenes"]), 1)
        self.assertEqual(len(narrative["curve_points"]), 1)
        self.assertEqual([line["id"] for line in narrative["story_lines"]], ["protagonist_action"])
        joined = "\n".join(preview["warnings"])
        self.assertIn("scenes 跳过 2", joined)
        self.assertIn("curve_points 跳过 1", joined)
        self.assertIn("story_lines 跳过 1", joined)

    def test_import_on_project_without_completed_shots_raises(self):
        """检测未完成的项目（全是待分析尾段）导入应明确报错，不能越界崩溃。"""
        empty = Session()
        self.addCleanup(empty.close_current_store)
        empty.build_index(str(Path(self.directory.name) / "clip.mkv"))
        empty.new_project()  # 单镜头 pending_start=0，没有已完成镜头
        with self.assertRaisesRegex(ValueError, "没有已完成镜头"):
            lapian.normalize_analysis(empty, {"scenes": [{"start_shot": 1, "end_shot": 1}]})

    def test_compare_isolates_bad_paths_and_dedupes_self(self):
        self.call("lapian.import", text=json.dumps(model_payload(self.session), ensure_ascii=False))
        mine = Path(self.directory.name) / "mine.shotlab"
        self.session.save_as(str(mine))
        self.session.close_current_store()
        reopened = Session()
        self.addCleanup(reopened.close_current_store)
        reopened.open_project(str(mine))
        result = METHODS["narrative.compare"](reopened, {
            "others": [str(mine),  # 重复自身 → 去重
                       str(Path(self.directory.name) / "不存在.shotlab")]})
        self.assertEqual(len(result["films"]), 2)
        self.assertIn("error", result["films"][1])
        self.assertTrue(result["films"][0]["has_narrative"])

    def test_deep_import_survives_junk_cards(self):
        self.call("lapian.import", text=json.dumps(model_payload(self.session), ensure_ascii=False))
        scene = self.call("narrative.get")["scenes"][0]
        reply = {"scene_id": scene["id"], "shot_cards": ["垃圾", {"shot": 1, "action": "红场"}]}
        result = self.call("lapian.deep_import", text=json.dumps(reply, ensure_ascii=False))
        self.assertEqual(result["shot_cards"], 1)
        self.assertIn("跳过 1", "\n".join(result["warnings"]))

    def test_package_single_image_and_frames_mode_files(self):
        out = Path(self.directory.name) / "pkg-one"
        result = lapian.export_package(self.session, {"out_dir": str(out), "mode": "frames",
                                                      "max_images": 1})
        self.assertEqual(result["images"], 1)
        self.assertEqual(len(list((out / "frames").glob("*.jpg"))), 1)

    def test_extract_json_tolerates_escaped_braces_and_quotes(self):
        text = '说明 {"a": "含\\"引号\\\"与{括号}"} 结束'
        self.assertEqual(lapian.extract_json(text), {"a": '含"引号"与{括号}'})

    def test_nan_and_infinity_never_reach_the_narrative_layer(self):
        """Python json 收下 NaN/Infinity 字面量；写进层文件会让 Rust serde 拒载整个项目。"""
        raw = ('{"filmIdentity": null, "scenes": [{"start_shot": 1, "end_shot": 3, "title": "场", '
               '"confidence": NaN}], "curve_points": ['
               '{"time": NaN, "intensity": NaN, "valence": Infinity, "title": "坏点"},'
               '{"time": Infinity, "intensity": 40, "title": "超时点"},'
               '{"time": 0.5, "intensity": 55, "title": "好点"}]}')
        self.call("lapian.import", text=raw)
        narrative = self.call("narrative.get")
        # 严格序列化不允许 NaN/Infinity——能通过即证明层里没有非有限值
        json.dumps(narrative, allow_nan=False)
        times = [p["time"] for p in narrative["curve_points"]]
        self.assertEqual(times, [0.5])  # NaN/Infinity 时间点均按无效丢弃并计入警告
        values = [p["intensity"] for p in narrative["curve_points"]]
        self.assertTrue(all(v is not None and v == v for v in values))
        self.assertIsNone(narrative["scenes"][0]["confidence"])  # NaN 置空而非传染
        self.assertEqual(lapian.parse_time("nan"), None)
        self.assertEqual(lapian.parse_time(float("inf")), None)

    def test_import_rollback_when_note_write_fails(self):
        """write_notes 中途失败（如笔记写入未完成镜头）必须整体回滚叙事层。"""
        payload = model_payload(self.session)
        # 场景范围合法但把镜头表换成 pending 尾段之外的手工场景——用 monkeypatch 让
        # set_note 在第二个镜头抛错，验证回滚。
        original = self.session.project.set_note
        calls = []

        def flaky(shot_id, text):
            calls.append(shot_id)
            if len(calls) > 1:
                raise ValueError("模拟笔记写入失败")
            original(shot_id, text)
        self.session.project.set_note = flaky
        try:
            with self.assertRaisesRegex(ValueError, "模拟笔记写入失败"):
                self.call("lapian.import", text=json.dumps(payload, ensure_ascii=False),
                          write_notes=True)
        finally:
            self.session.project.set_note = original
        self.assertEqual(self.call("narrative.get")["scenes"], [])  # 叙事层已回滚
        undo = self.session.history.undo
        self.assertFalse(any(e.label == "导入 AI 拉片" for e in undo))  # 没有半截记录


if __name__ == "__main__":
    unittest.main()
