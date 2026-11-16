"""Regression tests for research v2: interchange, evidence, analysis and statistics."""

import json
import csv
import threading
import unittest
from concurrent.futures import CancelledError
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from shotlab import study, study_formats as formats
from shotlab.rpc import METHODS, Session
from tests.test_motion_slit import run_job, session_with_shots


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.session = session_with_shots(self.directory.name)
        self.addCleanup(self.session.close_current_store)
        METHODS["study.template"](self.session, {})
        self.shots = self.session.state()["shots"]

    def call(self, method, **params):
        return METHODS[method](self.session, params)

    def relation(self, **extra):
        defaults = dict(relation_type="shot_reverse", transition_type="cut",
                        source_shot_id=self.shots[0]["id"], target_shot_id=self.shots[1]["id"],
                        frame=13, explanation="视线相对", author="A")
        return self.call("study.relation.put", **dict(defaults, **extra))

    def test_backward_compatible_defaults_and_undoable_new_records(self):
        self.assertIn("evidence_cards", self.session.state()["study"])
        old = {"version": 1, "tiers": [], "events": [], "exposures": [], "sound_reviews": {},
               "subtitles_imported": False}
        self.session.study = old
        state = METHODS["project.state"](self.session, {})
        self.assertEqual(state["study"]["version"], 2)
        self.assertEqual(state["study"]["shot_relations"], [])
        self.relation()
        relation = self.session.state()["study"]["shot_relations"][0]
        METHODS["edit.undo"](self.session, {})
        self.assertEqual(self.session.study["shot_relations"], [])
        METHODS["edit.redo"](self.session, {})
        self.assertEqual(self.session.study["shot_relations"][0]["id"], relation["id"])
        self.assertTrue(self.call("study.relation.report")["relations"][0]["active"])
        with self.assertRaises(ValueError):
            self.relation(relation_type="并不存在")

    def test_eaf_json_roundtrip_refuses_overwrite_and_preserves_manual_revision(self):
        self.event = self.call("study.event.put", tier_id=self.session.study["tiers"][0]["id"],
                               start_frame=0, end_frame=20, label="开场", author="A")
        self.relation()
        job = run_job(self.session, "study.exposure.run", {"start_frame": 0, "end_frame": 12})
        self.assertEqual(job["phase"], "done", job)
        sheet = self.session.study["exposures"][0]
        self.call("study.exposure.edit", id=sheet["id"], operation="split", frame=6, author="校正者")
        target = Path(self.directory.name) / "bundle.json"
        result = self.call("study.export", path=str(target))
        self.assertTrue(Path(result["eaf"]).exists())
        root = ET.parse(result["eaf"]).getroot()
        self.assertEqual(root.get("VERSION"), "3.0")
        self.assertIn("不自动授予", root.findtext("LICENSE") or "")
        self.assertTrue(root.findall("TIER/ANNOTATION/ALIGNABLE_ANNOTATION/ANNOTATION_VALUE"))
        self.assertTrue(root.findall("CONTROLLED_VOCABULARY/CV_ENTRY_ML/CVE_VALUE"))
        self.assertTrue(root.findall("LANGUAGE"))
        self.assertTrue(root.findall("TIME_ORDER/TIME_SLOT"))
        self.assertTrue(root.findall("TIER/ANNOTATION/ALIGNABLE_ANNOTATION"))
        link = root.find("REF_LINK_SET/CROSS_REF_LINK")
        self.assertEqual((link is not None, link is not None and bool(link.get("REF1")),
                          link is not None and bool(link.get("REF2"))), (True, True, True))
        self.assertEqual(len(root.findall("REF_LINK_SET/CROSS_REF_LINK")), 1)
        self.assertGreater(Path(result["relations"]).stat().st_size, 0)
        self.assertIn("source_version", json.loads(target.read_text(encoding="utf-8"))["provenance"])
        with self.assertRaises(ValueError):
            self.call("study.export", path=str(target))

        from tests.test_rpc import json_free_result
        from tests.test_research import _split
        other = Session()
        self.addCleanup(other.close_current_store)
        other.build_index(self.session.index.path)
        other.new_project()
        other.apply_detection(json_free_result(self.directory.name))
        _split(other, 12, 36)
        eaf_path = result["eaf"]
        result = METHODS["study.import_eaf"](other, {"path": eaf_path})
        self.assertEqual(result["study_import"]["format"], "EAF 3.0 + CadensView projection")
        self.assertEqual(len(other.study["shot_relations"]), 1)
        self.assertEqual(other.study["exposures"][0]["boundaries"], [0, 6, 12])
        before = deepcopy(other.study)
        second = METHODS["study.import_eaf"](other, {"path": eaf_path})
        self.assertEqual(second["study_import"]["imported"], 0)
        self.assertEqual(before, other.study)
        self.assertEqual(METHODS["study.import_json"](other, {"path": str(target)})["study_import"]["imported"], 0)
        root.find(".//ANNOTATION_VALUE").text = "外部修订"
        ET.SubElement(root.find("TIME_ORDER"), "TIME_SLOT", TIME_SLOT_ID="edited_start", TIME_VALUE="125")
        root.find(".//ALIGNABLE_ANNOTATION").set("TIME_SLOT_REF1", "edited_start")
        edited = target.with_name("edited.eaf")
        ET.ElementTree(root).write(edited, encoding="utf-8")
        conflict = METHODS["study.import_eaf"](other, {"path": str(edited)})["study_import"]
        self.assertEqual((conflict["imported"], conflict["conflicts"]), (0, 1))
        self.assertEqual(other.study["events"][0]["label"], "开场")
        METHODS["study.event.remove"](other, {"id": other.study["events"][0]["id"]})
        METHODS["study.import_eaf"](other, {"path": str(edited)})
        self.assertEqual(other.study["events"][0]["label"], "外部修订")
        self.assertEqual(other.study["events"][0]["start_frame"], 3)
        self.assertEqual(other.study["events"][0]["review_status"], "unreviewed")

    def test_generic_eaf_import_maps_milliseconds_to_source_frames(self):
        from shotlab.study_formats import eaf_xml
        session = self.session
        payload = {
            "version": 2, "tiers": [dict(id="t", name="对白", kind="dialogue", vocabulary=[], definition="", version=1)],
            "events": [dict(id="e", tier_id="t", start_frame=0, end_frame=12, label="你好", note="",
                            speaker="", author="", source="manual", review_status="confirmed",
                            created_at="", updated_at="", template_version=1)],
            "exposures": [], "sound_reviews": {}, "shot_relations": [], "evidence_cards": [],
            "adjudications": [], "speaker_links": [], "color_observations": [],
            "version_alignments": [], "saved_queries": [], "subtitles_imported": False,
        }
        xml = eaf_xml(session, payload)
        xml = xml.replace('NAME="cadensview-json"', 'NAME="not-a-bundle"')
        path = Path(self.directory.name) / "generic.eaf"
        path.write_text(xml, encoding="utf-8")
        result = self.call("study.import_eaf", path=str(path))
        self.assertIn("generic tiers", result["study_import"]["format"])
        self.assertEqual(result["study_import"]["imported"], 1)
        event = self.session.study["events"][0]
        self.assertLess(event["start_frame"], event["end_frame"])
        self.assertLessEqual(event["end_frame"], session.project.frame_count)
        before = len(self.session.history.undo)
        self.assertEqual(self.call("study.import_eaf", path=str(path))["study_import"]["imported"], 0)
        self.assertEqual(len(self.session.study["events"]), 1)
        self.assertEqual(len(self.session.history.undo), before)

    def test_invalid_json_import_preserves_working_state(self):
        tier = self.session.study["tiers"][0]["id"]
        before = deepcopy(self.session.study)
        undo = len(self.session.history.undo)
        path = Path(self.directory.name) / "bad.json"
        for bundle in (
            {"study": {"events": [{"id": "bad", "tier_id": tier}], "subtitles_imported": True}},
            {"study": {}, "schema_version": 999},
            {"study": {}, "source": {"sampled_sha256": "different source"}},
        ):
            path.write_text(json.dumps(bundle), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.call("study.import_json", path=str(path))
            self.assertEqual(self.session.state()["study"], before)
            self.assertEqual(len(self.session.history.undo), undo)

    def test_difference_export_phase_template_and_cancel_are_safe(self):
        run_job(self.session, "study.exposure.run", {"start_frame": 0, "end_frame": 10})
        sheet = self.session.study["exposures"][0]
        self.call("study.exposure.edit", id=sheet["id"], operation="template", template="classic")
        self.assertEqual(len(self.session.study["exposures"][0]["phases"]), 4)
        self.call("study.exposure.edit", id=sheet["id"], operation="split", frame=5)
        target = Path(self.directory.name) / "frames" / "exposure.png"
        result = run_job(self.session, "study.difference.run", {"sheet_id": sheet["id"], "path": str(target)})
        self.assertEqual(result["phase"], "done", result)
        payload = result["result"]
        manifest = json.loads(Path(payload["path"]).read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["frames"]), 10)
        self.assertEqual(manifest["boundaries"], [0, 5, 10])
        self.assertEqual(len(list(Path(payload["frames"]).glob("*.png"))), 10)
        self.assertTrue(Path(payload["difference"]).is_file())
        with Image.open(payload["difference"]) as image:
            self.assertEqual(image.mode, "L")
        duplicate = run_job(self.session, "study.difference.run", {"sheet_id": sheet["id"], "path": str(target)})
        self.assertEqual(duplicate["phase"], "error")
        self.assertIn("已存在", duplicate.get("message", ""))
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            study.difference_run(self.session, {"sheet_id": sheet["id"],
                                                "path": str(Path(self.directory.name) / "cancelled.png")}, cancel)
        self.assertFalse((Path(self.directory.name) / "cancelled_frames").exists())

    def test_difference_export_does_not_delete_racing_export(self):
        sheet = study.exposure_run(self.session, {"start_frame": 0, "end_frame": 4})["sheet_id"]
        root = Path(self.directory.name).resolve()
        folder = root / "race_frames"
        sentinel = folder / "other-job.txt"
        mkdir = Path.mkdir

        def competing_mkdir(path, *args, **kwargs):
            if path == folder:
                mkdir(folder)
                sentinel.write_text("other job", encoding="utf-8")
            return mkdir(path, *args, **kwargs)

        with patch.object(Path, "mkdir", competing_mkdir), self.assertRaises(FileExistsError):
            study.difference_run(self.session, {"sheet_id": sheet, "path": str(root / "race.png")})
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "other job")

    def test_relation_export_uses_the_same_live_boundary_check(self):
        self.relation()
        self.session._edit("移动边界", mutate=lambda: self.session.project.move_boundary(1, 13))
        self.assertFalse(self.call("study.relation.report")["relations"][0]["active"])
        outputs = self.call("study.export", path=str(Path(self.directory.name) / "moved.json"))
        with open(outputs["relations"], encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(next(csv.DictReader(stream))["active"], "False")

    def test_evidence_capture_cannot_be_attached_to_another_frame(self):
        target = Path(self.directory.name) / "evidence.png"
        result = run_job(self.session, "study.evidence.capture",
                         {"project_id": self.session.project.id, "frame": 14, "path": str(target)})
        self.assertEqual(result["phase"], "done", result)
        self.assertEqual(result["result"]["frame"], 14)
        with Image.open(target) as image:
            self.assertEqual(image.format, "PNG")
        with self.assertRaises(ValueError):
            self.call("study.evidence.put", frame=15, attachment_frame=14, attachment_path=str(target))
        self.call("study.evidence.put", frame=14, attachment_frame=14, attachment_path=str(target))
        self.assertEqual(self.session.study["evidence_cards"][0]["attachment_frame"], 14)

    def test_evidence_disagreement_and_adjudication_keep_original_opinions(self):
        common = dict(frame=14, documented="镜头表第 3 号", dialogue="", note="", bibliography="分镜稿 p.2",
                      author="A", reviewer="")
        self.call("study.evidence.put", interpretation="视线匹配", **common)
        self.call("study.evidence.put", interpretation="视线不匹配", author="B", **{
            k: v for k, v in common.items() if k != "author"})
        report = self.call("study.evidence.report")
        self.assertEqual(len(report["disagreements"]), 1)
        self.call("study.evidence.adjudicate", frame=14, decision="以 B 为准", reason="角色视线向画右",
                  reviewer="C", author="A")
        cards = self.call("study.evidence.report")["cards"]
        self.assertEqual(len({c["interpretation"] for c in cards}), 2)
        self.assertEqual(self.session.study["adjudications"][0]["decision"], "以 B 为准")
        with self.assertRaises(ValueError):
            self.call("study.evidence.put", frame=14, kind="invalid")

    def test_spectrogram_sound_segments_no_dialogue_and_optional_asr(self):
        from shotlab.audio import SAMPLE_RATE, Soundtrack
        pcm = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        pcm[:SAMPLE_RATE] = .2 * np.sin(np.arange(SAMPLE_RATE) * 2 * np.pi * 440 / SAMPLE_RATE)
        raw = dict(pcm=pcm, momentary=np.array([-20.0]), short_term=np.array([-20.0]), fingerprint=np.array([]))
        track = Soundtrack(raw, .25)
        self.session.soundtrack = track
        self.session.soundtrack_key = (self.session.index.path, self.session.index.size, self.session.index.mtime_ns)
        result = run_job(self.session, "study.waveform.run", {})
        self.assertEqual(result["phase"], "done", result)
        self.assertTrue(result["result"]["spectrogram"]["columns"])
        self.assertTrue(result["result"]["segments"])
        tier = next(t["id"] for t in self.session.study["tiers"] if t["kind"] == "dialogue")
        self.call("study.event.put", tier_id=tier, start_frame=0, end_frame=12, label="台词")
        report = self.call("study.no_dialogue", start_frame=0, end_frame=48)
        self.assertEqual(len(report["shots"]), 3)
        self.assertEqual(len(report["scenes"]), 0)
        self.assertIn("未标注", report["note"])
        self.assertEqual(report["range"]["no_dialogue_s"], report["shots"][0]["no_dialogue_s"] +
                         report["shots"][1]["no_dialogue_s"] + report["shots"][2]["no_dialogue_s"])
        self.assertFalse(self.call("study.asr_status", adapter="whisperx")["available"])
        with self.assertRaises(ValueError):
            self.call("study.asr_status", adapter="whisperx", segments=[{"start_s": 0, "end_s": 1}])
        imported = self.call("study.asr_status", adapter="whisperx",
                             segments=[{"start_s": 0, "end_s": 1, "text": "你好"}],
                             model="whisperx", model_version="test", source="人工转录", author="A")
        self.assertEqual((imported["available"], imported["status"], imported["model_version"]),
                         (True, "user_provided", "test"))
        with self.assertRaises(ValueError):
            self.call("study.asr_status", adapter="unknown")
        self.call("study.speaker.put", start_frame=0, end_frame=12, speaker="旁白", character="角色 A",
                  shot_id=self.shots[0]["id"], position="offscreen", author="A")
        self.assertEqual(self.session.study["speaker_links"][0]["position"], "offscreen")

    def test_unreviewed_subtitles_do_not_verify_a_scene(self):
        tiers = {tier["kind"]: tier["id"] for tier in self.session.study["tiers"]}
        self.call("study.event.put", tier_id=tiers["dialogue"], start_frame=0, end_frame=6,
                  label="字幕", review_status="unreviewed")
        self.call("study.event.put", tier_id=tiers["scene"], start_frame=0, end_frame=12, label="开场")
        scene = self.call("study.no_dialogue", start_frame=0, end_frame=12)["scenes"][0]
        self.assertFalse(scene["reviewed"])
        self.assertGreater(scene["duration_s"], scene["no_dialogue_s"])
        self.assertFalse(self.call("study.no_dialogue", start_frame=0, end_frame=6)["range"]["reviewed"])

    def test_color_roi_masks_composition_and_provenance(self):
        result = run_job(self.session, "study.color.run", {
            "start_frame": 0, "end_frame": 12, "frame": 4,
            "roi": [0, 0, 1, 1], "exclusions": [[0, 0, 1, .25], [0, .75, 1, .25]],
            "subject": [.25, .25, .25, .5]})
        self.assertEqual(result["phase"], "done", result)
        self.assertGreater(result["result"]["color"]["pixels"], 0)
        self.assertGreaterEqual(result["result"]["color"]["luminance"], 0)
        self.assertIn("zone", result["result"]["composition"]["subject"])
        with self.assertRaises(ValueError):
            self.call("study.color.put", start_frame=0, end_frame=12, roi=[0, 0, 1, 1],
                      exclusions=[[0, .9, 1, .1]], subject=[.25, .25, .25, .5], observation=result["result"])
        with self.assertRaises(ValueError):
            self.call("study.color.put", start_frame=12, end_frame=36, roi=[0, 0, 1, 1],
                      exclusions=result["result"]["exclusions"], subject=[.25, .25, .25, .5],
                      observation=result["result"])
        self.call("study.color.put", start_frame=0, end_frame=12, label="冷色", scene="开场",
                  phase="预备", character="角色 A", roi=[0, 0, 1, 1], exclusions=result["result"]["exclusions"],
                  subject=[.25, .25, .25, .5], gaze="右", observation=result["result"],
                  judgment="人物偏右，需与第二镜头对照", author="A")
        observation = self.session.study["color_observations"][0]
        self.assertEqual(observation["subject"], [.25, .25, .25, .5])
        self.assertIn("pixels", observation["observation"]["color"])
        self.assertEqual(observation["observation"]["frame"], 4)
        self.assertNotEqual(observation["observation"], observation["judgment"])

    def test_saved_query_alignment_clipshots_and_equal_weight(self):
        self.call("study.query.put", name="长镜头", clauses=[{"field": "duration_min", "value": .4}], author="A")
        query = self.session.study["saved_queries"][0]
        result = self.call("study.query.run", id=query["id"])
        self.assertEqual(len(result["shots"]), len(self.shots))
        self.assertIn("kinds", result["shots"][0])
        self.relation()
        self.call("study.alignment.put", name="不同版本", first_label="电视版", second_label="蓝光版",
                  version_note="蓝光版有裁剪", anchors=[[1, 1, "composition_similarity", "构图相似"],
                                                        [3, 4, "material_reuse", "OP 复用"]], author="A")
        alignment = self.session.study["version_alignments"][0]["report"]
        self.assertEqual((alignment["composition_similarity"], alignment["material_reuse"]), (1, 1))
        self.assertEqual(alignment["anchors"][1]["inserted_first"], 1)
        with self.assertRaises(ValueError):
            formats.manual_alignment([[2, 2, "composition_similarity"], [1, 3, "composition_similarity"]])
        metrics = self.call("study.validate_reference", references={"a": [[10, 11], [20, 24]]},
                            predictions={"a": [[10, 11], [20, 24]]}, source="ClipShots 格式参考集")
        self.assertEqual(metrics["cut"]["f1"], 1.0)
        self.assertEqual(metrics["gradual"]["f1"], 1.0)
        equal = formats.equal_weight_summary([dict(label="A", durations=[1, 2, 3]),
                                               dict(label="B", durations=[10] * 200)], "film")
        self.assertAlmostEqual(equal["mean_of_film_means_s"], 6.0)
        self.assertTrue(equal["per_film_equal_weight"])

    def test_reference_metrics_match_half_open_intervals_once(self):
        metrics = formats.clipshots_metrics({"a": [[10, 11], [11, 12]]},
                                            {"a": [[10, 11]], "extra": [[8, 9]]})
        self.assertEqual(metrics["cut"]["matched"], 1)
        self.assertEqual(metrics["cut"]["f1"], .5)
        adjacent = formats.clipshots_metrics({"a": [[10, 11]]}, {"a": [[11, 12]]})
        self.assertEqual(adjacent["cut"]["matched"], 0)
        tolerant = formats.clipshots_metrics({"a": [[10, 11]]}, {"a": [[11, 12]]}, tolerance=1)
        self.assertEqual(tolerant["cut"]["matched"], 1)
        greedy_trap = formats.clipshots_metrics({"a": [[0, 10], [5, 7]]}, {"a": [[0, 7], [0, 4]]})
        self.assertEqual(greedy_trap["gradual"]["matched"], 2)

    def test_vlm_evidence_frames_and_research_method_versions(self):
        from shotlab.vlm import DEFAULTS, annotate_shots
        from tests.test_research import FakeBackend
        labels, errors, meta = annotate_shots(self.session.project, self.session.index, existing={},
                                              config=dict(DEFAULTS, frames=3), backend=FakeBackend())
        self.assertEqual(errors, [])
        self.assertTrue(all(label["evidence_frames"] for label in labels.values()))
        self.assertTrue(all(len(label["evidence_times_s"]) == label["evidence_count"] for label in labels.values()))
        self.assertEqual(formats.adaptive_frame_count(5, 3), 3)
        self.assertEqual(formats.adaptive_frame_count(2400, 3), 6)
        self.assertIn("adaptive", meta["evidence_sampling"])
        target = Path(self.directory.name) / "research.csv"
        result = self.call("research.export", path=str(target))
        method = json.loads(Path(result["method"]).read_text(encoding="utf-8"))
        self.assertEqual(method["study_schema_version"], 2)
        self.assertEqual(method["software_version"], study.SOFTWARE_VERSION)
        self.assertIn("provenance", method)
        header = target.read_text(encoding="utf-8-sig").splitlines()[0]
        self.assertIn("vlm_evidence_frames", header)
        self.assertIn("vlm_evidence_times_s", header)

    def test_compare_equal_weight_and_manual_alignment(self):
        from shotlab.compare import compare_projects
        with TemporaryDirectory() as folder:
            paths = []
            for number, cuts in enumerate(((12, 36), (6, 12, 18, 24, 30, 36, 42))):
                session = Session() if number else self.session
                if number:
                    from tests.test_rpc import video, json_free_result
                    from tests.test_research import _split
                    session.build_index(str(video(Path(folder) / f"v{number}.mkv")))
                    session.new_project()
                    session.apply_detection(json_free_result(folder))
                    _split(session, *cuts)
                path = Path(folder) / f"p{number}.shotlab"
                METHODS["project.save_as"](session, {"path": str(path),
                                                    "overwrite": number == 0 and self.session.project_path == str(Path(path).resolve())})
                if number == 0:
                    self.session.close_current_store()
                else:
                    session.close_current_store()
                paths.append(str(path))
            # Reopen the first project to write a saved query into it, then
            # persist through the ordinary project save path.
            self.session.store = None
            self.session.project_path = None
            METHODS["project.open"](self.session, {"path": paths[0]})
            result = compare_projects(paths, inference_unit="episode",
                                      alignment=[[1, 1, "composition_similarity"], [2, 3, "material_reuse"]])
            self.assertEqual(result["equal_weight"]["inference_unit"], "episode")
            self.assertTrue(result["equal_weight"]["per_film_equal_weight"])
            self.assertEqual(len(result["equal_weight"]["films"]), 2)
            self.assertEqual(result["version_alignment"]["material_reuse"], 1)
            self.assertIsNone(result["pairs"][0]["mann_whitney"])
            self.assertEqual(result["pairs"][0]["first_n"], 1)
            self.call("regions.add", label="OP", start_frame=0, end_frame=12)
            scene_tier = next(t["id"] for t in self.session.study["tiers"] if t["kind"] == "scene")
            self.call("study.event.put", tier_id=scene_tier, start_frame=12, end_frame=36, label="同名")
            self.call("study.event.put", tier_id=scene_tier, start_frame=36, end_frame=48, label="同名")
            METHODS["project.save"](self.session, {})
            self.call("study.query.put", name="跨集长镜头",
                      clauses=[{"field": "duration_min", "value": .4}], author="A")
            query = next(q for q in self.session.study["saved_queries"] if q["name"] == "跨集长镜头")
            filtered = self.call("compare.run", paths=paths, inference_unit="episode", saved_query_id=query["id"])
            self.assertEqual(filtered["query_summary"]["total"], sum(row["query_total"] for row in filtered["films"]))
            self.assertEqual([r["query_total"] for r in filtered["films"]], [2, 8])
            self.assertEqual([r["shot_count"] for r in filtered["films"]], [2, 0])
            self.assertIsNotNone(filtered["query_summary"]["strata"]["mean_of_film_means_s"])
            self.assertTrue(all("query_matched" in row for row in filtered["films"]))
            self.assertTrue(filtered["equal_weight"]["explicit"])
            self.assertTrue(filtered["equal_weight"]["strata"])
            compared = METHODS["compare.run"](self.session, {"paths": paths, "inference_unit": "episode"})
            self.assertEqual(compared["equal_weight"]["inference_unit"], "episode")
            self.assertIsNone(compared["query_summary"])
            scenes = compare_projects(paths, inference_unit="scene")
            self.assertEqual((scenes["pairs"][0]["first_n"], scenes["pairs"][0]["second_n"]), (2, 0))
            shots = compare_projects(paths, inference_unit="shot")
            self.assertEqual((shots["pairs"][0]["first_n"], shots["pairs"][0]["second_n"]), (2, 8))
            self.assertIn("p_holm", shots["pairs"][0]["mann_whitney"])
            self.session.close_current_store()
            with self.assertRaises(ValueError):
                compare_projects(paths, inference_unit="episode", query=[{"field": "not-a-field"}])
            with self.assertRaises(ValueError):
                compare_projects(paths, inference_unit="frame")


if __name__ == "__main__":
    unittest.main()
