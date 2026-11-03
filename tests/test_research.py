"""计量电影学 / 动画研究扩展：稳健统计、拍数、标注层、审校、比较、重复镜头、VLM。"""

import errno
import http.server
import json
import threading
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import av
import numpy as np

from shotlab import rpc as rpc_module
from shotlab.rpc import METHODS, Session
from shotlab.statistics.robust import cliffs_delta, cohen_kappa, mann_whitney, median_ci, poisson_ci, summary
from tests.anime_fixtures import _write, anime_clip
from tests.test_motion_slit import run_job, session_with_shots
from tests.test_rpc import json_free_result


def _split(session, *frames):
    for frame in frames:
        session._edit("拆分", mutate=lambda frame=frame: session.project.split(frame))


class RobustStatisticsTests(unittest.TestCase):
    def test_median_interval_brackets_the_median(self):
        values = sorted(float(v) for v in range(1, 101))
        low, high = median_ci(values)
        self.assertLess(low, 50.5)
        self.assertGreater(high, 50.5)
        self.assertIsNone(median_ci([1.0, 2.0]))

    def test_poisson_interval_matches_exact_values_closely(self):
        # 精确 Garwood 区间（k=10）：3.17–18.39
        low, high = poisson_ci(10)
        self.assertAlmostEqual(low, 4.795, delta=0.1)
        self.assertAlmostEqual(high, 18.39, delta=0.1)
        self.assertEqual(poisson_ci(0)[0], 0.0)

    def test_dominance_and_rank_test(self):
        short, long = [1, 2, 3, 4, 5] * 6, [6, 7, 8, 9, 10] * 6
        self.assertEqual(cliffs_delta(long, short)["delta"], 1.0)
        self.assertEqual(cliffs_delta(long, short)["magnitude"], "large")
        self.assertLess(mann_whitney(short, long)["p_two_sided"], 1e-6)
        self.assertAlmostEqual(mann_whitney(short, short)["p_two_sided"], 1.0)

    def test_kappa(self):
        self.assertEqual(cohen_kappa([("CS", "CS"), ("LS", "LS"), ("MS", "MS")])["kappa"], 1.0)
        result = cohen_kappa([("CS", "LS"), ("LS", "CS")])
        self.assertLess(result["kappa"], 0)
        self.assertEqual(cohen_kappa([(None, "CS")])["n"], 0)

    def test_summary_reports_rate_with_interval(self):
        data = summary([2.0] * 30)
        self.assertAlmostEqual(data["cutting_rate_per_min"], 30.0)
        low, high = data["cutting_rate_ci95_per_min"]
        self.assertLess(low, 30.0)
        self.assertGreater(high, 30.0)
        self.assertEqual(data["iqr_s"], 0.0)


class StatsOverviewTests(unittest.TestCase):
    def test_overview_carries_msl_interval_and_cutting_rate(self):
        with TemporaryDirectory() as directory:
            session = session_with_shots(directory)
            overview = METHODS["stats.run"](session, {})["overview"]
            self.assertIn("msl_ci95_s", overview)
            # 夹具时间基 1/1000，48 帧时长约 1.999s
            self.assertAlmostEqual(overview["cutting_rate_per_min"], 3 / (48 / 24) * 60, delta=0.1)
            self.assertEqual(len(overview["cutting_rate_ci95_per_min"]), 2)


class KomaTests(unittest.TestCase):
    def test_holds_and_pans_are_separated(self):
        with TemporaryDirectory() as directory:
            session = Session()
            session.build_index(str(anime_clip(Path(directory) / "anime.mkv")))
            session.new_project()
            session.apply_detection(json_free_result(directory))
            _split(session, 36, 72, 96)
            result = session.motion_run(session.analysis_scope({}))
            rows = {row["shot_number"]: row for row in result["shots"]}
            self.assertEqual([rows[n]["koma"] for n in (1, 2, 3)], [3, 2, 1])
            self.assertTrue(all(rows[n]["koma_reliable"] for n in (1, 2, 3)))
            self.assertAlmostEqual(rows[1]["drawings_per_s"], 8.0, delta=0.5)
            self.assertEqual(rows[4]["pan_share"], 1.0)
            self.assertEqual(rows[4]["drawings"], 1)
            self.assertFalse(rows[4]["koma_reliable"])
            self.assertEqual(result["timing_summary"]["shots_with_reliable_koma"], 3)

    def test_rate_warning_for_telecine_sources(self):
        from shotlab.animation import rate_warning
        self.assertIn("IVTC", rate_warning("30000/1001"))
        self.assertIsNone(rate_warning("24000/1001"))
        self.assertIsNone(rate_warning("bad"))


class AnnotationLayerTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.session = session_with_shots(self.directory.name)
        self.addCleanup(self.session.close_current_store)
        self.ids = [s["id"] for s in self.session.state()["shots"]]

    def test_tags_cut_numbers_and_manual_scale_are_undoable(self):
        session = self.session
        METHODS["tags.set"](session, {"shot_id": self.ids[0], "tags": ["夜景"]})
        METHODS["cut.set"](session, {"shot_id": self.ids[0], "value": "012"})
        METHODS["scale.set_manual"](session, {"shot_id": self.ids[1], "label": "CS"})
        state = session.state()
        self.assertEqual(state["shots"][0]["cut_number"], "012")
        self.assertEqual(state["shots"][1]["scale_manual"], "CS")
        METHODS["edit.undo"](session, {})
        self.assertIsNone(session.state()["shots"][1]["scale_manual"])
        METHODS["edit.undo"](session, {})
        self.assertEqual(session.state()["shots"][0]["cut_number"], "")
        METHODS["edit.undo"](session, {})
        self.assertEqual(session.state()["shots"][0]["tags"], [])
        METHODS["edit.redo"](session, {})
        self.assertEqual(session.state()["shots"][0]["tags"], ["夜景"])
        with self.assertRaises(ValueError):
            METHODS["scale.set_manual"](session, {"shot_id": self.ids[0], "label": "huge"})

    def test_regions_exclude_shots_from_statistics(self):
        session = self.session
        METHODS["regions.add"](session, {"label": "OP", "start_frame": 0, "end_frame": 12})
        self.assertEqual(session.state()["shots"][0]["region"], "OP")
        stats = METHODS["stats.run"](session, {"exclude_regions": True})
        self.assertEqual(stats["overview"]["shot_count"], 2)
        self.assertEqual(METHODS["stats.run"](session, {})["overview"]["shot_count"], 3)
        region_id = session.regions[0]["id"]
        METHODS["regions.remove"](session, {"id": region_id})
        self.assertEqual(session.regions, [])
        METHODS["edit.undo"](session, {})
        self.assertEqual(len(session.regions), 1)
        with self.assertRaises(ValueError):
            METHODS["regions.add"](session, {"label": "OP", "start_frame": 10, "end_frame": 5})

    def test_layers_persist_and_orphans_are_pruned_on_save(self):
        session = self.session
        METHODS["tags.set"](session, {"shot_id": self.ids[2], "tags": ["孤儿"]})
        METHODS["metadata.set"](session, {"values": {"title": "测试片", "episode": "3"}})
        METHODS["regions.add"](session, {"label": "ED", "start_frame": 36, "end_frame": 48})
        METHODS["edit.merge"](session, {"row": 1})  # 第 3 个镜头并入第 2 个，ID 消失
        path = Path(self.directory.name) / "p.shotlab"
        METHODS["project.save_as"](session, {"path": str(path)})
        reopened = Session()
        try:
            reopened.open_project(str(path))
            self.assertEqual(reopened.metadata["title"], "测试片")
            self.assertEqual(reopened.regions[0]["label"], "ED")
            self.assertNotIn(self.ids[2], reopened.shot_tags)
        finally:
            reopened.close_current_store()
        with self.assertRaises(ValueError):
            METHODS["metadata.set"](session, {"values": {"bogus": "x"}})

    def test_save_as_without_hard_links_still_refuses_to_overwrite(self):
        session = self.session
        target = Path(self.directory.name) / "fat.shotlab"
        with mock.patch("os.link", side_effect=OSError(errno.EPERM, "no hard links")):
            METHODS["project.save_as"](session, {"path": str(target)})
            self.assertTrue(target.exists())
            staged = Path(self.directory.name) / "staged"
            staged.write_bytes(b"x")
            with self.assertRaises(FileExistsError):
                rpc_module._publish_new(staged, target)
            self.assertTrue(staged.exists())

    def test_srt_position_spec_is_ignored(self):
        srt = Path(self.directory.name) / "pos.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000 X1:10 X2:20 Y1:5 Y2:9\n台词\n", encoding="utf-8")
        result = METHODS["subtitles.load"](self.session, {"path": str(srt)})
        self.assertEqual(result["cues"], 1)
        self.assertEqual(result["entries"][0]["end_us"], 1_000_000)


class ReviewAndAuditTests(unittest.TestCase):
    def test_review_queue_flags_low_scores_short_shots_and_near_misses(self):
        with TemporaryDirectory() as directory:
            session = Session()
            from tests.test_rpc import video
            session.build_index(str(video(Path(directory) / "clip.mkv")))
            session.new_project()
            result_dir = Path(directory) / "det"
            result_dir.mkdir()
            (result_dir / "result.json").write_text(json.dumps({
                "boundaries": [{"frame": 12, "score": 0.55}, {"frame": 24, "score": 0.98}],
                "score_file": "scores.npz", "parameters": {"threshold": 0.5}}), encoding="utf-8")
            single = np.zeros(48, np.float32)
            single[11], single[23], single[39] = 0.55, 0.98, 0.31  # 39 → 帧 40 的疑似漏检
            np.savez_compressed(result_dir / "scores.npz", single=single, many=single)
            session.apply_detection(str(result_dir))
            self.assertEqual(session.detection_info["near_misses"], [[40, 0.31]])
            queue = METHODS["review.run"](session, {})
            frames = {item["frame"]: item for item in queue["items"]}
            self.assertIn(12, frames)
            self.assertTrue(any("置信度低" in r for r in frames[12]["reasons"]))
            self.assertEqual(frames[40]["kind"], "missed")
            METHODS["review.confirm"](session, {"frame": 12})
            self.assertTrue(METHODS["review.run"](session, {})["items"][-1]["confirmed"])
            # 人工把 24 挪到 26、在 40 新增切点：审计应计入“移动”和“新增”
            session._edit("移动", mutate=lambda: session.project.move_boundary(2, 26))
            _split(session, 40)
            audit = METHODS["research.audit"](session, {})
            self.assertEqual((audit["kept"], audit["moved"], audit["added"], audit["removed"]), (1, 1, 1, 0))
            self.assertAlmostEqual(audit["manual_change_ratio"], 1.0)

    def test_research_export_writes_table_method_and_lengths(self):
        with TemporaryDirectory() as directory:
            session = session_with_shots(directory)
            METHODS["metadata.set"](session, {"values": {"title": "片", "series": "系列"}})
            session.motion_run(session.analysis_scope({}))
            out = Path(directory) / "research.csv"
            result = METHODS["research.export"](session, {"path": str(out)})
            self.assertEqual(result["written"], 3)
            header = out.read_text(encoding="utf-8-sig").splitlines()[0]
            for column in ("cut_number", "koma", "vlm_scale", "region", "scale_manual"):
                self.assertIn(column, header)
            method = json.loads(Path(result["method"]).read_text(encoding="utf-8"))
            self.assertEqual(method["metadata"]["series"], "系列")
            self.assertEqual(method["summary"]["shot_count"], 3)
            self.assertEqual(len(Path(result["lengths"]).read_text().split()), 3)


def textured(seed, count=12, size=(54, 96)):
    rng = np.random.default_rng(seed)
    block = rng.integers(0, 255, (size[0] // 6, size[1] // 6, 3), dtype=np.uint8)
    frame = np.repeat(np.repeat(block, 6, axis=0), 6, axis=1)
    return [frame.copy() for _ in range(count)]


class RepeatsAndCompareTests(unittest.TestCase):
    def test_bank_shot_is_found_and_neighbours_are_ignored(self):
        with TemporaryDirectory() as directory:
            clip = _write(Path(directory) / "bank.mkv", textured(1) + textured(2) + textured(1) + textured(3))
            session = Session()
            session.build_index(str(clip))
            session.new_project()
            session.apply_detection(json_free_result(directory))
            _split(session, 12, 24, 36)
            event = run_job(session, "repeats.run", {})
            self.assertEqual(event["phase"], "done", event)
            groups = event["result"]["groups"]
            self.assertEqual(len(groups), 1)
            self.assertEqual([s["shot_number"] for s in groups[0]["shots"]], [1, 3])
            self.assertEqual(METHODS["repeats.get"](session, {})["groups"], groups)

    def test_compare_two_projects(self):
        with TemporaryDirectory() as directory:
            paths = []
            for n, cuts in enumerate(((12, 36), (6, 12, 18, 24, 30, 36, 42))):
                session = session_with_shots(directory) if n == 0 else None
                if session is None:
                    session = Session()
                    from tests.test_rpc import video
                    session.build_index(str(video(Path(directory) / f"clip{n}.mkv")))
                    session.new_project()
                    session.apply_detection(json_free_result(directory))
                    _split(session, *cuts)
                METHODS["metadata.set"](session, {"values": {"title": f"片{n}", "episode": str(n + 1)}})
                path = Path(directory) / f"p{n}.shotlab"
                METHODS["project.save_as"](session, {"path": str(path)})
                session.close_current_store()
                paths.append(str(path))
            result = METHODS["compare.run"](Session(), {"paths": paths, "csv_path": str(Path(directory) / "c.csv")})
            self.assertEqual([f["shot_count"] for f in result["films"]], [3, 8])
            pair = result["pairs"][0]
            self.assertGreater(pair["cliffs_delta"]["delta"], 0)  # 第一部镜头更长
            self.assertIn("p_holm", pair["mann_whitney"])
            self.assertEqual([t["episode"] for t in result["episode_trend"]], ["1", "2"])
            self.assertTrue(Path(result["csv"]).exists())
            with self.assertRaises(ValueError):
                METHODS["compare.run"](Session(), {"paths": paths[:1]})


class FakeBackend:
    def __init__(self, segment="none"):
        self.calls = 0
        self.segment = segment

    def annotate(self, shot, images):
        self.calls += 1
        assert images and all(isinstance(i, str) for i in images)
        return dict(shot_scale="CS", camera_movement="static", camera_angle="eye", subjects=1, setting="interior",
                    content_type="dialogue", special_segment=self.segment, effects=[], description="对话",
                    confidence=0.8), "fake-model"

    def retryable(self, error):
        return False


class VlmTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.session = session_with_shots(self.directory.name)

    def test_labels_are_cached_per_boundary_and_model(self):
        from shotlab.vlm import DEFAULTS, annotate_shots
        session = self.session
        backend = FakeBackend()
        labels, errors, meta = annotate_shots(session.project, session.index, existing={},
                                              config=dict(DEFAULTS), backend=backend)
        self.assertEqual((len(labels), errors, backend.calls), (3, [], 3))
        session.vlm_labels = labels
        again, _, _ = annotate_shots(session.project, session.index, existing=labels,
                                     config=dict(DEFAULTS), backend=backend)
        self.assertEqual((again, backend.calls), ({}, 3))
        self.assertEqual(session.state()["shots"][0]["vlm"]["shot_scale"], "CS")
        session._edit("移动", mutate=lambda: session.project.move_boundary(1, 10))
        self.assertIsNone(session.state()["shots"][0]["vlm"])  # 边界变了，旧标注失效

    def test_vlm_job_and_region_suggestions(self):
        session = self.session
        with mock.patch("shotlab.vlm.backend_for", return_value=FakeBackend("op")):
            event = run_job(session, "vlm.run", {})
        self.assertEqual(event["phase"], "done", event)
        self.assertEqual(event["result"]["annotated"], 3)
        suggestions = METHODS["vlm.suggest_regions"](session, {})["suggestions"]
        self.assertEqual(suggestions[0]["label"], "OP")
        self.assertEqual((suggestions[0]["start_frame"], suggestions[0]["end_frame"]), (0, 48))
        METHODS["scale.set_manual"](session, {"shot_id": session.project.shots[0].id, "label": "CS"})
        agreement = METHODS["scale.agreement"](session, {})
        self.assertEqual(agreement["vlm_vs_manual"]["n"], 1)

    def test_reply_parsing_is_strict_about_enums(self):
        from shotlab.vlm import parse_reply
        parsed = parse_reply('```json\n{"shot_scale": "CS", "camera_movement": "orbit", "subjects": "2", '
                             '"effects": ["glow", "bogus"], "confidence": 7}\n```')
        self.assertEqual(parsed["shot_scale"], "CS")
        self.assertIsNone(parsed["camera_movement"])
        self.assertEqual((parsed["subjects"], parsed["effects"], parsed["confidence"]), (2, ["glow"], 1.0))
        with self.assertRaises(ValueError):
            parse_reply("no json here")

    def test_openai_compatible_backend_over_http(self):
        from shotlab.statistics import build_context
        from shotlab.vlm import DEFAULTS, OpenAICompatibleBackend
        seen = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                seen["path"] = self.path
                seen["auth"] = self.headers.get("Authorization")
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen["images"] = sum(1 for part in body["messages"][1]["content"] if part["type"] == "image_url")
                reply = json.dumps({"model": "qwen-vl", "choices": [{"message": {"content": json.dumps(
                    {"shot_scale": "LS", "camera_movement": "pan", "camera_angle": "high", "subjects": 0,
                     "setting": "exterior", "content_type": "establishing", "special_segment": "none",
                     "effects": ["none"], "description": "远景", "confidence": 0.6})}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        config = dict(DEFAULTS, provider="openai_compatible", model="qwen-vl",
                      base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="test-key")
        shot = build_context(self.session.project, self.session.index).records[0]
        label, model = OpenAICompatibleBackend(config).annotate(shot, ["aGVsbG8="] * 3)
        self.assertEqual((label["shot_scale"], label["camera_movement"], model), ("LS", "pan", "qwen-vl"))
        self.assertEqual((seen["path"], seen["auth"], seen["images"]), ("/v1/chat/completions", "Bearer test-key", 3))

    def test_anthropic_backend_request_shape(self):
        from types import SimpleNamespace
        from shotlab.statistics import build_context
        from shotlab.vlm import DEFAULTS, SCHEMA, AnthropicBackend
        captured = {}
        reply = json.dumps({"shot_scale": "MS", "camera_movement": "static", "camera_angle": "eye", "subjects": 2,
                            "setting": "interior", "content_type": "dialogue", "special_segment": "none",
                            "effects": [], "description": "两人对话", "confidence": 0.9})

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(stop_reason="end_turn", model=kwargs["model"],
                                   content=[SimpleNamespace(type="text", text=reply)])

        backend = AnthropicBackend(dict(DEFAULTS, api_key="test"))
        backend.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
        shot = build_context(self.session.project, self.session.index).records[0]
        label, model = backend.annotate(shot, ["aGVsbG8="] * 2)
        self.assertEqual((label["shot_scale"], model), ("MS", "claude-opus-5-5"))
        self.assertEqual(captured["output_config"]["format"], {"type": "json_schema", "schema": SCHEMA})
        self.assertEqual((captured["betas"], captured["fallbacks"]), (["server-side-fallback-2026-07-01"], "default"))
        images = [part for part in captured["messages"][0]["content"] if part["type"] == "image"]
        self.assertEqual(len(images), 2)
        self.assertNotIn("temperature", captured)

        def refuse(**kwargs):
            return SimpleNamespace(stop_reason="refusal", model=kwargs["model"], content=[])
        backend.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=refuse)))
        with self.assertRaises(RuntimeError):
            backend.annotate(shot, ["aGVsbG8="])

    def test_config_never_returns_the_key(self):
        from shotlab import vlm
        with TemporaryDirectory() as home:
            with mock.patch.object(vlm, "CONFIG_PATH", Path(home) / "vlm.json"):
                saved = vlm.save_config({"provider": "openai_compatible", "base_url": "http://x/v1",
                                         "model": "glm-4v", "api_key": "secret"})
                self.assertTrue(saved["has_key"])
                self.assertNotIn("api_key", saved)
                self.assertNotIn("secret", json.dumps(METHODS["vlm.config.get"](self.session, {})))
                with self.assertRaises(ValueError):
                    vlm.save_config({"provider": "openai_compatible", "base_url": ""})


class ExportJobTests(unittest.TestCase):
    def test_sheets_and_stills_run_as_cancellable_jobs(self):
        with TemporaryDirectory() as directory:
            session = session_with_shots(directory)
            pdf = Path(directory) / "sheet.pdf"
            event = run_job(session, "export.pdf_sheet", {"path": str(pdf)})
            self.assertEqual(event["phase"], "done", event)
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
            folder = Path(directory) / "stills"
            self.assertEqual(run_job(session, "export.stills", {"path": str(folder), "count": 4})["phase"], "done")
            again = run_job(session, "export.stills", {"path": str(folder), "count": 4})
            self.assertEqual((again["phase"], again["kind"]), ("error", "FileExistsError"))

    def test_select_is_allowed_while_a_job_runs(self):
        self.assertIn("edit.select", rpc_module.BUSY_ALLOWED)


class CliTimeTests(unittest.TestCase):
    def test_cli_seconds_are_relative_to_the_first_frame(self):
        from shotlab.cli import _relative_seconds
        with TemporaryDirectory() as directory:
            session = session_with_shots(directory)
            index = replace(session.index, pts=[p + 900 for p in session.index.pts])
            self.assertEqual(_relative_seconds(index, 0), 0.0)
            self.assertAlmostEqual(_relative_seconds(index, 24), float(Fraction(session.index.pts[24]) * index.tb))


if __name__ == "__main__":
    unittest.main()
