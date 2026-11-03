"""Research workflow: cut review queue, detector-vs-manual audit, long-table export.

Automatic cut detection always has false and missed cuts; a study must say how
many boundaries were checked and changed by hand. The queue ranks boundaries by
evidence that they may be wrong; the audit compares the current partition with
the detector's output saved at apply time.
"""

import csv
import json
from concurrent.futures import CancelledError
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from shotlab.statistics import build_context
from shotlab.statistics.robust import cohen_kappa, summary
from t0.media import FrameReader

LOW_SCORE = 0.7          # TransNetV2 boundary score below which a cut is worth a look
SHORT_FRAMES = 6         # shots this short are often split by flashes or effects
SMALL_CHANGE = 0.08      # mean absolute difference (0-1) across the cut
NEAR_MISS_SCORE = 0.25   # detector peaks below the 0.5 threshold that may be missed cuts
MOVED_FRAMES = 3         # a manual boundary within this many frames of a detected one was moved
SCALES = ("ECS", "CS", "MS", "FS", "LS")


def detection_record(result, scores_path=None, offset=0):
    """What to keep from a detector result.json/scores.npz for later auditing.

    `offset` is the first frame the detector analysed: score k describes the
    transition into frame offset + k + 1, the same rule its boundaries use.
    """
    cuts = {str(int(b["frame"])): b.get("score") for b in result.get("boundaries", [])}
    near = []
    if scores_path and Path(scores_path).exists():
        with np.load(scores_path) as scores:
            single = np.asarray(scores["single"], dtype=np.float32) if "single" in scores else None
        if single is not None:
            for i in range(1, len(single) - 1):
                v = float(single[i])
                if NEAR_MISS_SCORE <= v < 0.5 and v >= single[i - 1] and v >= single[i + 1]:
                    near.append([int(offset + i + 1), round(v, 4)])
    return dict(detector=result.get("detector") or result.get("name") or "transnetv2",
                parameters=result.get("parameters", {}), cuts=cuts, near_misses=near,
                applied_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def edit_audit(project, detection):
    """Detected cuts vs the current completed partition (frame-exact, with a small move window)."""
    completed_end = project.boundaries[project.completed_count]
    current = set(project.boundaries[1:project.completed_count])
    if not detection or detection.get("cuts") is None:
        return dict(available=False, current_cuts=len(current))
    detected = {int(f) for f in detection["cuts"] if 0 < int(f) < completed_end}
    kept = current & detected
    removed = sorted(detected - current)
    added = sorted(current - detected)
    moved = []
    for frame in list(added):
        near = [d for d in removed if abs(d - frame) <= MOVED_FRAMES]
        if near:
            original = min(near, key=lambda d: abs(d - frame))
            moved.append([original, frame])
            removed.remove(original)
            added.remove(frame)
    changed = len(removed) + len(added) + len(moved)
    return dict(available=True, detected_cuts=len(detected), current_cuts=len(current), kept=len(kept),
                removed=len(removed), added=len(added), moved=len(moved),
                manual_change_ratio=changed / len(detected) if detected else None,
                precision_vs_final=len(kept) / len(detected) if detected else None,
                recall_vs_final=len(kept) / len(current) if current else None,
                removed_frames=removed, added_frames=added, moved_frames=moved)


def _small(frame):
    array = frame.reformat(width=64, height=36, format="rgb24").to_ndarray() if frame.width > 64 \
        else frame.to_ndarray(format="rgb24")
    return array.astype(np.float32) / 255.0


def review_queue(project, index, detection, confirmed=(), cancel=None, progress=None):
    """Rank boundaries (and near-miss frames) by reasons to re-check them by eye."""
    detection = detection or {}
    scores = {int(k): v for k, v in (detection.get("cuts") or {}).items()}
    confirmed = {int(f) for f in confirmed}
    count = project.completed_count
    boundaries = project.boundaries[1:count]
    items = []
    reader = FrameReader(index, sequential=True)
    try:
        for done, frame in enumerate(boundaries, 1):
            if cancel is not None and cancel.is_set():
                raise CancelledError("审校队列已取消")
            row = project.boundaries.index(frame)
            left = frame - project.boundaries[row - 1]
            right = project.boundaries[row + 1] - frame
            before, after = reader.read_range(frame - 1, frame + 1)
            change = float(np.abs(_small(before) - _small(after)).mean())
            score = scores.get(frame)
            reasons, weight = [], 0.0
            if detection.get("cuts") is not None and frame not in scores:
                reasons.append("人工新增"); weight += 0.5
            if score is not None and score < LOW_SCORE:
                reasons.append(f"检测置信度低 {score:.2f}"); weight += (LOW_SCORE - score) * 2
            if min(left, right) <= SHORT_FRAMES:
                reasons.append(f"相邻镜头仅 {min(left, right)} 帧"); weight += 1.0
            if change < SMALL_CHANGE:
                reasons.append(f"切点两侧画面差异小 {change:.3f}"); weight += (SMALL_CHANGE - change) * 10
            items.append(dict(kind="cut", frame=frame, boundary=row, score=score, change=round(change, 4),
                              left_frames=left, right_frames=right, reasons=reasons,
                              priority=round(weight, 3), confirmed=frame in confirmed))
            if progress is not None:
                progress(done, len(boundaries) + 1)
    finally:
        reader.close()
    current = set(boundaries)
    completed_end = project.boundaries[count]
    for frame, score in detection.get("near_misses") or []:
        if 0 < frame < completed_end and not any(abs(frame - c) <= MOVED_FRAMES for c in current):
            items.append(dict(kind="missed", frame=int(frame), score=score, reasons=[f"疑似漏检（峰值 {score:.2f}）"],
                              priority=round(score * 2, 3), confirmed=int(frame) in confirmed))
    flagged = [i for i in items if i["reasons"]]
    flagged.sort(key=lambda i: (i["confirmed"], -i["priority"], i["frame"]))
    return dict(items=flagged, checked=len(boundaries), flagged=len(flagged),
                confirmed=sum(1 for i in flagged if i["confirmed"]),
                thresholds=dict(low_score=LOW_SCORE, short_frames=SHORT_FRAMES, small_change=SMALL_CHANGE,
                                near_miss_score=NEAR_MISS_SCORE))


def scale_agreement(rows):
    """Cohen's kappa of model and VLM scale labels against manual labels."""
    model = cohen_kappa([(r.get("scale_manual"), (r.get("scale") or {}).get("label")) for r in rows])
    vlm = cohen_kappa([(r.get("scale_manual"), (r.get("vlm") or {}).get("shot_scale")) for r in rows])
    return dict(model_vs_manual=model, vlm_vs_manual=vlm,
                note="kappa 以人工标注为参照；样本少于 50 时只作参考")


def _blank(value):
    return "" if value is None else value


def research_rows(session, params):
    """One row per completed Shot in scope, with every annotation layer side by side."""
    state = session.state()
    rows = session.scoped_shots(params)
    meta = session.metadata
    motion = {}
    if session.motion is not None:
        motion = {row["shot_id"]: row for row in session.motion.get("shots", [])}
    sound, cut_sound = {}, {}
    if session.audio is not None:
        sound = {row["shot_id"]: row for row in session.audio.get("shots", [])}
        cut_sound = {cut["right_shot_id"]: cut for cut in session.audio.get("cuts", [])}
    region_of = session.region_lookup()
    out = []
    for shot in rows:
        timing = motion.get(shot["id"], {})
        loud, cut = sound.get(shot["id"], {}), cut_sound.get(shot["id"], {})
        # 声音结果只在镜头边界未变时沿用；边界改过的镜头留空，等重新分析。
        if (loud.get("start_frame"), loud.get("end_frame_exclusive")) != (shot["start"], shot["end"]):
            loud = {}
        if cut.get("frame") != shot["start"]:
            cut = {}
        vlm = shot.get("vlm") or {}
        scale = shot.get("scale") or {}
        out.append({
            "title": meta.get("title", ""), "series": meta.get("series", ""), "episode": meta.get("episode", ""),
            "shot_number": shot["number"] + 1, "shot_id": shot["id"], "cut_number": shot.get("cut_number", ""),
            "start_frame": shot["start"], "end_frame_exclusive": shot["end"],
            "frame_count": shot["end"] - shot["start"],
            "start_s": round(session.index.us(shot["start"]) / 1e6 - session.index.us(0) / 1e6, 6),
            "duration_s": shot["duration_s"], "region": region_of(shot["start"], shot["end"]) or "",
            "tags": "; ".join(shot.get("tags", [])), "scale_manual": shot.get("scale_manual") or "",
            "scale_model": scale.get("label", ""), "scale_model_confidence": scale.get("confidence", ""),
            "vlm_scale": vlm.get("shot_scale", ""), "vlm_camera": vlm.get("camera_movement", ""),
            "vlm_content": vlm.get("content_type", ""), "vlm_description": vlm.get("description", ""),
            "motion": timing.get("motion", ""), "koma": timing.get("koma_label") or "",
            "koma_reliable": timing.get("koma_reliable", ""), "drawings_per_s": timing.get("drawings_per_s", ""),
            "still_share": timing.get("still_share", ""), "pan_share": timing.get("pan_share", ""),
            "drawn_motion": timing.get("drawn_motion", ""),
            "loudness_lufs": _blank(loud.get("loudness_lufs")), "loudness_max_lufs": _blank(loud.get("loudness_max_lufs")),
            "lra_lu": _blank(loud.get("lra_lu")), "silence_share": _blank(loud.get("silence_share")),
            "cut_in_silence": _blank(cut.get("in_silence")), "cut_loudness_jump_lu": _blank(cut.get("loudness_jump_lu")),
            "cut_sound_relation": cut.get("relation", ""), "cut_sound_offset_s": _blank(cut.get("sound_offset_s")),
            "note": shot.get("note", ""),
        })
    return out, state


def export_research(session, params):
    """Long-table CSV + method sidecar JSON (+ plain shot-length list) for statistics software."""
    path = Path(params["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, state = research_rows(session, params)
    if not rows:
        raise ValueError("没有可导出的已完成镜头")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lengths = path.with_suffix(".lengths.txt")
    lengths.write_text("\n".join(f"{row['duration_s']:.3f}" for row in rows) + "\n", encoding="utf-8")
    index = session.index
    scope = session.analysis_scope(params)
    context = build_context(session.project, index, scope)
    method = dict(
        software="CadensView", exported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=dict(path=index.path, size=index.size, frame_count=len(index.pts), time_base=index.time_base,
                    average_rate=index.metadata.get("average_rate"), width=index.metadata.get("width"),
                    height=index.metadata.get("height")),
        metadata=session.metadata, scope=context.scope, warnings=context.warnings,
        regions=session.regions, excluded_regions=bool(params.get("exclude_regions")),
        detection=dict((session.detection_info or {}).get("parameters", {}),
                       detector=(session.detection_info or {}).get("detector")),
        edit_audit=edit_audit(session.project, session.detection_info),
        summary=summary([row["duration_s"] for row in rows]),
        annotation_layers=dict(
            scale_model="MobileNetV3 trained on live-action MovieShots; unvalidated on animation",
            scale_manual="human annotation in CadensView",
            vlm=(session.vlm_meta or {}) or None,
            motion=(session.motion or {}).get("parameters"),
            audio=(session.audio or {}).get("parameters")),
        columns=list(rows[0]))
    sidecar = path.with_suffix(".method.json")
    sidecar.write_text(json.dumps(method, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"path": str(path.resolve()), "method": str(sidecar.resolve()), "lengths": str(lengths.resolve()),
            "written": len(rows)}
