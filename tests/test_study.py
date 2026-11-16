"""Research workflows: source-time invariants, review provenance and persistence."""

from copy import deepcopy
from concurrent.futures import CancelledError
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest

import numpy as np

from shotlab import study
from shotlab.rpc import METHODS, Session
from shotlab.vlm import normalize
from tests.test_motion_slit import session_with_shots, run_job
from tests.anime_fixtures import _write, held_square
from tests.test_audio import _project


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.session = session_with_shots(self.directory.name)
        self.addCleanup(self.session.close_current_store)
        self.call("study.template")

    def call(self, method_name, **params):
        return METHODS[method_name](self.session, params)

    def tier(self, kind):
        return next(t["id"] for t in self.session.study["tiers"] if t["kind"] == kind)

    def event(self, kind="scene", start=4, end=40, label="对话场景", **extra):
        self.call("study.event.put", tier_id=self.tier(kind), start_frame=start, end_frame=end, label=label, **extra)
        return self.session.study["events"][-1]

    def test_cross_cut_annotations_survive_edits_undo_and_save(self):
        event = deepcopy(self.event(author="研究者 A"))
        self.call("edit.merge", row=0)
        self.call("edit.split", frame=9)
        self.assertEqual(self.session.study["events"], [event])
        self.call("edit.undo")
        self.call("edit.undo")
        self.assertEqual(self.session.study["events"], [event])
        self.call("edit.undo")
        self.assertEqual(self.session.study["events"], [])
        self.call("edit.redo")
        path = Path(self.directory.name) / "research.shotlab"
        self.call("project.save_as", path=str(path))
        other = Session()
        self.addCleanup(other.close_current_store)
        other.open_project(str(path))
        self.assertEqual(other.study, self.session.study)

    def test_invalid_edits_are_atomic_and_tiers_do_not_orphan_events(self):
        self.event()
        old = deepcopy(self.session.study)
        for a, b in [(-1, 5), (4, 4), (6, 5), (0, 49), (1.5, 5), (True, 4)]:
            with self.assertRaises(ValueError):
                self.event(start=a, end=b)
            self.assertEqual(old, self.session.study)
        with self.assertRaises(ValueError):
            self.call("study.tier.remove", id=self.tier("scene"))
        with self.assertRaises(ValueError):
            self.call("study.tier.put", id=self.tier("scene"), name="场景", kind="music")
        self.assertEqual(old, self.session.study)
        self.call("study.template")
        self.assertEqual(len(self.session.study["tiers"]), 6)

    def test_vocabulary_and_review_provenance(self):
        self.call("study.tier.put", id=self.tier("action"), name="动作阶段", vocabulary=["预备", "动作", "缓冲"])
        with self.assertRaises(ValueError):
            self.event(kind="action", label="不存在")
        event = self.event(kind="action", label="预备", author="A")
        self.call("study.event.put", id=event["id"], label="动作", author="B", review_status="revised")
        edited = self.session.study["events"][-1]
        self.assertEqual((edited["source"], edited["author"], edited["review_status"]), ("manual", "B", "revised"))
        self.assertEqual(edited["created_at"], event["created_at"])

    def test_coverage_unions_overlapping_dialogue_and_keeps_music_independent(self):
        self.event(kind="dialogue", start=0, end=24, label="A")
        self.event(kind="dialogue", start=12, end=36, label="B")
        self.event(kind="music", start=0, end=48, label="配乐")
        coverage = study.coverage(self.session, 0, 48)
        self.assertAlmostEqual(coverage["dialogue_share"], .75, delta=.001)
        self.assertAlmostEqual(coverage["music_share"], 1)
        self.assertAlmostEqual(study.coverage(self.session, 36, 48)["dialogue_share"], 0)

    def test_subtitles_import_is_idempotent_and_edited_text_drives_playback(self):
        path = Path(self.directory.name) / "dialogue.srt"
        path.write_text("1\n00:00:00,010 --> 00:00:00,900\n你好\n\n2\n00:00:00,800 --> 00:00:01,500\n回答\n", encoding="utf-8")
        self.call("subtitles.load", path=str(path))
        self.call("study.import_subtitles")
        events = self.session.study["events"]
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["start_frame"], 0)
        self.assertEqual(events[0]["review_status"], "unreviewed")
        self.call("study.event.put", id=events[0]["id"], label="校正台词", review_status="confirmed")
        self.call("study.import_subtitles")
        self.assertEqual(len(self.session.study["events"]), 2)
        self.assertEqual(self.session.state()["subtitle_cues"][0]["text"], "校正台词")
        project = Path(self.directory.name) / "subtitles.shotlab"
        self.call("project.save_as", path=str(project))
        other = Session()
        self.addCleanup(other.close_current_store)
        other.open_project(str(project))
        self.assertEqual(other.state()["subtitle_cues"], self.session.state()["subtitle_cues"])

    def test_subtitle_mapping_uses_source_pts_not_average_rate(self):
        # Deliberately nonzero origin and VFR spacing, with a known end point.
        index = self.session.index
        index.time_base = "1/1000"
        index.pts = [5000 + f * f * 2 for f in range(48)]
        index.durations = [index.pts[i + 1] - index.pts[i] for i in range(47)] + [190]
        self.session.subtitle_cues = [dict(start_us=15_000, end_us=72_000, text="VFR")]
        self.call("study.import_subtitles")
        event = self.session.study["events"][0]
        self.assertEqual((event["start_frame"], event["end_frame"]), (2, 6))
        cue = self.session.state()["subtitle_cues"][0]
        self.assertEqual((cue["start_us"], cue["end_us"]), (8000, 72000))

    def test_exposure_analysis_correction_and_cancel(self):
        result = run_job(self.session, "study.exposure.run", {"start_frame": 0, "end_frame": 12, "roi": [0, 0, 1, 1]})
        self.assertEqual(result["phase"], "done", result)
        sheet = self.session.study["exposures"][0]
        identity = sheet["id"]
        self.assertEqual(sheet["boundaries"], [0, 12])
        self.call("study.exposure.edit", id=identity, operation="split", frame=6)
        sheet = self.session.study["exposures"][0]
        self.assertEqual(sheet["boundaries"], [0, 6, 12])
        self.assertEqual(sheet["estimated_boundaries"], [0, 12])
        self.call("study.exposure.edit", id=identity, operation="confirm", author="A")
        self.assertEqual(self.session.study["exposures"][0]["review_status"], "confirmed")
        self.call("edit.undo")
        self.assertEqual(self.session.study["exposures"][0]["review_status"], "revised")
        for frame in (0, 12):
            with self.assertRaises(ValueError):
                self.call("study.exposure.edit", id=identity, operation="join", frame=frame)
        before = deepcopy(self.session.study)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            study.exposure_run(self.session, {"start_frame": 0, "end_frame": 12}, cancel)
        self.assertEqual(before, self.session.study)
        for roi in ([0, 0, 0, 1], [.8, 0, .3, 1], [float("nan"), 0, 1, 1]):
            with self.assertRaises(ValueError):
                study.validate_roi(roi)

    def test_blank_exposure_name_uses_default_title(self):
        study.exposure_run(self.session, {"start_frame": 0, "end_frame": 12, "name": "  "})
        self.assertEqual(self.session.study["exposures"][0]["name"], "可见曝光估计")

    def test_roi_keeps_independent_animation_layers_separate(self):
        top, bottom = held_square(3, 24), held_square(1, 24)
        frames = [np.concatenate((a, b), axis=0) for a, b in zip(top, bottom)]
        path = _write(Path(self.directory.name) / "layers.mkv", frames)
        session = _project(self.directory.name, path)
        self.addCleanup(session.close_current_store)
        for y in (0, .5):
            study.exposure_run(session, dict(start_frame=0, end_frame=24, roi=[0, y, 1, .5]))
        slow, fast = session.study["exposures"]
        self.assertEqual(slow["boundaries"], list(range(0, 25, 3)))
        self.assertEqual(fast["boundaries"], list(range(25)))

    def test_waveform_preserves_peaks_time_offset_and_last_bin(self):
        from shotlab.audio import Soundtrack, SAMPLE_RATE
        pcm = np.zeros(12001, dtype=np.float32)
        pcm[0], pcm[1], pcm[-1] = -.8, .7, .9
        raw = dict(pcm=pcm, momentary=np.array([]), short_term=np.array([]), fingerprint=np.array([]))
        track = Soundtrack(raw, .25)
        self.assertLessEqual(len(track.waveform), 6000)
        self.assertEqual(track.waveform[0]["time_s"], .25)
        self.assertAlmostEqual(track.waveform[0]["low"], -.8)
        self.assertAlmostEqual(track.waveform[0]["high"], .7)
        self.assertAlmostEqual(track.waveform[-1]["high"], .9)
        self.assertAlmostEqual(track.waveform[-1]["end_s"], .25 + len(pcm) / SAMPLE_RATE, places=5)

    def test_sound_reviews_invalidate_when_neighbouring_shot_changes(self):
        self.call("study.sound_review", frame=12, relation="J", author="A")
        self.assertEqual(study.sound_review(self.session, 12)["relation"], "J")
        self.call("edit.move_boundary", boundary=2, frame=35)
        self.assertEqual(study.sound_review(self.session, 12), {})
        self.call("edit.undo")
        self.assertEqual(study.sound_review(self.session, 12)["relation"], "J")

    def test_export_roundtrip_data_and_refuse_overwrites(self):
        self.event(author="A")
        target = Path(self.directory.name) / "bundle.json"
        self.call("study.export", path=str(target))
        bundle = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(bundle["study"], self.session.study)
        with target.with_suffix(".events.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
            self.assertEqual(row["label"], "对话场景")
            self.assertEqual(row["author"], "A")
        before = target.read_bytes()
        with self.assertRaises(ValueError):
            self.call("study.export", path=str(target))
        self.assertEqual(before, target.read_bytes())

    def test_unknown_vlm_fields_do_not_become_labels(self):
        label = normalize(dict(shot_scale="unknown", camera_movement="unknown", camera_angle="unknown"))
        self.assertIsNone(label["shot_scale"])
        self.assertIsNone(label["camera_movement"])


if __name__ == "__main__":
    unittest.main()
