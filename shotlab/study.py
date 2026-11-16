"""Source-anchored research annotations and editable visible-exposure estimates.

Intervals are [start_frame, end_frame); they survive Shot edits unchanged. All
mutations use Session._annotate, including analysis publication after cancel
checks. No generated result overwrites a manually corrected sheet.
"""

from bisect import bisect_left, bisect_right
from concurrent.futures import CancelledError
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from uuid import uuid4

from shotlab import study_formats as formats

KINDS = {"scene": "场景", "character": "角色", "dialogue": "对白", "music": "音乐",
         "camera": "运镜", "action": "动作阶段", "custom": "自定义"}
STATUSES = {"unreviewed", "revised", "confirmed"}
MAX_EXPOSURE_FRAMES = 5000
STUDY_VERSION = 2
SOFTWARE_VERSION = "0.1.0+research-2"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty():
    return dict(version=STUDY_VERSION, tiers=[], events=[], exposures=[], sound_reviews={}, subtitles_imported=False,
                shot_relations=[], evidence_cards=[], adjudications=[], speaker_links=[],
                color_observations=[], version_alignments=[], saved_queries=[], provenance=provenance())


def data(session):
    # Old project files contain schema v1 fields only. Return a complete v2
    # view without mutating the session itself, so the next undoable edit is
    # still a single atomic change to the original undo chain.
    value = {**empty(), **session.study}
    value["version"] = STUDY_VERSION
    value["provenance"] = {**provenance(session), **(session.study.get("provenance") or {}),
                           "study_schema_version": STUDY_VERSION,
                           "software_version": SOFTWARE_VERSION}
    return value


def provenance(session=None, extra=None):
    """Software/source/model versions for reproducible research exports."""
    result = dict(software="CadensView", software_version=SOFTWARE_VERSION, study_schema_version=STUDY_VERSION,
                  source_version="", source_id="", model="", served_by="", prompt_version=2,
                  bibliography="", author="", reviewer="", updated_at=now())
    if session is not None and session.index is not None:
        index = session.index
        result.update(source_version=str(index.mtime_ns), source_id=f"{index.size}:{index.mtime_ns}",
                      metadata=dict(session.metadata))
        vlm = getattr(session, "vlm_meta", None) or {}
        if vlm:
            result.update(model=str(vlm.get("model", "")), served_by=str(vlm.get("served_by", "")),
                          prompt_version=vlm.get("prompt_version", 2))
    if extra:
        result.update({k: formats.text(v, 500) for k, v in extra.items() if k in result})
    return result


def interval(session, params):
    if session.project is None or session.index is None:
        raise ValueError("先打开视频并建立索引")
    start, end = params.get("start_frame"), params.get("end_frame")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= session.project.frame_count:
        raise ValueError("帧范围须为 [起点, 排他终点)，且位于源视频内")
    return start, end


def item(items, identity):
    found = next((v for v in items if v["id"] == identity), None)
    if found is None:
        raise ValueError("记录不存在，请刷新后重试")
    return found


def commit(session, value, label):
    value["version"] = STUDY_VERSION
    value["provenance"] = dict(value.get("provenance") or provenance(session), updated_at=now())
    return session._annotate(label, lambda: setattr(session, "study", value))


def shot_at(session, frame):
    if session.project is None:
        raise ValueError("先打开视频并建立索引")
    for row, shot in enumerate(session.project.shots[:session.project.completed_count]):
        start, end = session.project.interval(row)
        if start <= frame < end:
            return dict(id=shot.id, number=row + 1, start_frame=start, end_frame=end)
    raise ValueError("该帧不在已完成镜头内")


def shot_by_id(session, identity):
    for row, shot in enumerate(session.project.shots[:session.project.completed_count]):
        if shot.id == identity:
            start, end = session.project.interval(row)
            return dict(id=shot.id, number=row + 1, start_frame=start, end_frame=end)
    raise ValueError("镜头不存在或尚未完成")


def tier_put(session, params, emit=None):
    value = deepcopy(data(session))
    old = item(value["tiers"], params["id"]) if params.get("id") else {}
    name = str(params.get("name", old.get("name", ""))).strip()
    kind = params.get("kind", old.get("kind", "custom"))
    vocabulary = params.get("vocabulary", old.get("vocabulary", []))
    if not name or kind not in KINDS or not isinstance(vocabulary, list) or any(not isinstance(x, str) for x in vocabulary):
        raise ValueError("填写层名称、合法类型和词表")
    if old and kind != old["kind"] and any(e["tier_id"] == old["id"] for e in value["events"]):
        raise ValueError("含标注的层不能改变类型，请新建一层")
    tier = dict(id=old.get("id", uuid4().hex), name=name[:120], kind=kind,
                vocabulary=list(dict.fromkeys(x.strip() for x in vocabulary if x.strip())),
                definition=str(params.get("definition", old.get("definition", "")))[:2000],
                version=int(old.get("version", 0)) + 1)
    if old:
        value["tiers"][value["tiers"].index(old)] = tier
    else:
        value["tiers"].append(tier)
    return commit(session, value, "编辑标注层")


def tier_remove(session, params, emit=None):
    value = deepcopy(data(session))
    target = item(value["tiers"], params["id"])
    if any(e["tier_id"] == target["id"] for e in value["events"]):
        raise ValueError("此层仍有标注，请先逐条删除或移到其他层")
    value["tiers"].remove(target)
    return commit(session, value, "删除空标注层")


def template(session, params, emit=None):
    value = deepcopy(data(session))
    existing = {t["kind"] for t in value["tiers"]}
    for kind, name in KINDS.items():
        if kind != "custom" and kind not in existing:
            value["tiers"].append(dict(id=uuid4().hex, name=name, kind=kind, vocabulary=[], definition="", version=1))
    return commit(session, value, "添加视听研究模板")


def event_put(session, params, emit=None):
    value = deepcopy(data(session))
    old = item(value["events"], params["id"]) if params.get("id") else {}
    merged = {**old, **params}
    start, end = interval(session, merged)
    tier = item(value["tiers"], merged.get("tier_id"))
    label = str(merged.get("label", "")).strip()
    status = merged.get("review_status", "revised" if old else "confirmed")
    if not label or status not in STATUSES:
        raise ValueError("填写标注内容和合法核对状态")
    if tier["vocabulary"] and label not in tier["vocabulary"]:
        raise ValueError("标注内容不在此层词表中")
    event = dict(old, id=old.get("id", uuid4().hex), tier_id=tier["id"], start_frame=start, end_frame=end,
                 label=label[:5000], speaker=str(merged.get("speaker", ""))[:200],
                 note=str(merged.get("note", ""))[:5000], author=str(merged.get("author", ""))[:200],
                 source=old.get("source", "manual"), review_status=status,
                 created_at=old.get("created_at", now()), updated_at=now(), template_version=tier["version"])
    if old:
        value["events"][value["events"].index(old)] = event
    else:
        value["events"].append(event)
    return commit(session, value, "编辑时间标注")


def event_remove(session, params, emit=None):
    value = deepcopy(data(session))
    value["events"].remove(item(value["events"], params["id"]))
    return commit(session, value, "删除时间标注")


def endpoint_us(index, frame):
    return (index.us(frame) if frame < len(index.pts) else index.end_us(len(index.pts) - 1)) - index.us(0)


def import_subtitles(session, params, emit=None):
    if session.index is None:
        raise ValueError("先打开视频")
    if not session.subtitle_cues:
        raise ValueError("先加载 SRT 字幕")
    value = deepcopy(data(session))
    tier = next((t for t in value["tiers"] if t["kind"] == "dialogue" and not t["vocabulary"]), None)
    if tier is None:
        tier = dict(id=uuid4().hex, name="对白", kind="dialogue", vocabulary=[], definition="字幕对白", version=1)
        value["tiers"].append(tier)
    times = [session.index.us(f) - session.index.us(0) for f in range(len(session.index.pts))]
    duration = endpoint_us(session.index, len(times))
    known = {e.get("import_key") for e in value["events"]}
    for cue in session.subtitle_cues:
        start_us, end_us = max(0, cue["start_us"]), min(duration, cue["end_us"])
        if end_us <= start_us:
            continue
        key = hashlib.sha256(json.dumps(cue, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if key in known:
            continue
        start = max(0, bisect_right(times, start_us) - 1)
        end = min(len(times), bisect_left(times, end_us))
        value["events"].append(dict(id=uuid4().hex, tier_id=tier["id"], start_frame=start,
                                    end_frame=max(start + 1, end), label=cue["text"], speaker="", note="", author="",
                                    source="subtitle", import_key=key, review_status="unreviewed",
                                    created_at=now(), updated_at=now(), template_version=tier["version"]))
        known.add(key)
    value["subtitles_imported"] = True
    return commit(session, value, "字幕导入对白层")


def playback_cues(session):
    value = data(session)
    if not value["subtitles_imported"] or session.index is None:
        return session.subtitle_cues
    tiers = {t["id"] for t in value["tiers"] if t["kind"] == "dialogue"}
    return sorted([dict(start_us=endpoint_us(session.index, e["start_frame"]),
                        end_us=endpoint_us(session.index, e["end_frame"]), text=e["label"])
                   for e in value["events"] if e["tier_id"] in tiers], key=lambda c: c["start_us"])


def coverage(session, start, end):
    """Union duration: overlapping speakers/tiers do not double count coverage."""
    value = data(session)
    kinds = {t["id"]: t["kind"] for t in value["tiers"]}
    result = {}
    total = endpoint_us(session.index, end) - endpoint_us(session.index, start)
    for kind in ("dialogue", "music"):
        spans = sorted((max(start, e["start_frame"]), min(end, e["end_frame"])) for e in value["events"]
                       if kinds.get(e["tier_id"]) == kind and e["start_frame"] < end and start < e["end_frame"])
        merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(b, merged[-1][1])
            else:
                merged.append([a, b])
        microseconds = sum(endpoint_us(session.index, b) - endpoint_us(session.index, a) for a, b in merged)
        result[kind + "_share"] = microseconds / total if total else None
    result["scenes"] = [e["label"] for e in value["events"] if kinds.get(e["tier_id"]) == "scene"
                        and e["start_frame"] < end and start < e["end_frame"]]
    return result


def validate_roi(roi):
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError("区域须为归一化的 x/y/宽/高")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in roi):
        raise ValueError("区域坐标须为有限数值")
    x, y, w, h = roi
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1.000001 or y + h > 1.000001:
        raise ValueError("区域须位于画面内且面积大于零")
    return [float(v) for v in roi]


def exposure_run(session, params, cancel=None, progress=None):
    import numpy as np
    from PIL import Image
    from t0.media import FrameReader
    from shotlab.animation import PairClassifier, rate_warning
    start, end = interval(session, params)
    if end - start > MAX_EXPOSURE_FRAMES:
        raise ValueError(f"单张曝光表最多 {MAX_EXPOSURE_FRAMES} 帧，请缩小范围")
    roi = validate_roi(params.get("roi", [0, 0, 1, 1]))
    classifier = PairClassifier()
    boundaries, changes = [start], []
    previous = None
    reader = FrameReader(session.index, sequential=True)
    try:
        for frame in range(start, end):
            if cancel is not None and cancel.is_set():
                raise CancelledError("曝光估计已取消")
            rgb = reader.read(frame).to_ndarray(format="rgb24")
            height, width = rgb.shape[:2]
            x, y, w, h = roi
            left, top = int(x * width), int(y * height)
            right, bottom = min(width, math.ceil((x + w) * width)), min(height, math.ceil((y + h) * height))
            if right - left < 2 or bottom - top < 2:
                raise ValueError("分析区域至少需要 2×2 个源像素")
            rgb = rgb[top:bottom, left:right]
            if rgb.shape[1] > 320:
                rgb = np.asarray(Image.fromarray(rgb).resize((320, max(2, round(rgb.shape[0] * 320 / rgb.shape[1])))))
            if previous is not None:
                pair = classifier.classify(previous, rgb)
                changes.append(round(pair["motion"], 5))
                if pair["kind"] in ("draw", "pan_draw"):
                    boundaries.append(frame)
            previous = rgb
            if progress is not None:
                progress(frame - start + 1, end - start)
    finally:
        reader.close()
    if cancel is not None and cancel.is_set():
        raise CancelledError("曝光估计已取消")
    boundaries.append(end)
    name = str(params.get("name") or "").strip() or "可见曝光估计"
    sheet = dict(id=uuid4().hex, name=name[:120],
                 start_frame=start, end_frame=end, roi=roi, boundaries=boundaries,
                 estimated_boundaries=list(boundaries), changes=changes, review_status="unreviewed",
                 phases=[], phase_template="",
                 difference_frames=sorted(range(start + 1, end), key=lambda f: changes[f - start - 1], reverse=True)[:12],
                 author=str(params.get("author", ""))[:200], created_at=now(), updated_at=now(),
                 method="frame-difference + global translation compensation; visible image changes, not production drawings",
                 warning=rate_warning(session.index.metadata.get("average_rate")))
    value = deepcopy(data(session))
    value["exposures"].append(sheet)
    commit(session, value, "新增曝光估计")
    return dict(sheet_id=sheet["id"], project_id=session.project.id)


def exposure_edit(session, params, emit=None):
    value = deepcopy(data(session))
    sheet = item(value["exposures"], params["id"])
    operation, frame = params.get("operation"), params.get("frame")
    if operation == "remove":
        value["exposures"].remove(sheet)
    elif operation in ("split", "join"):
        if type(frame) is not int or not sheet["start_frame"] < frame < sheet["end_frame"]:
            raise ValueError("只能修改曝光表内部的边界")
        if operation == "split":
            sheet["boundaries"] = sorted(set(sheet["boundaries"] + [frame]))
        elif frame not in sheet["boundaries"]:
            raise ValueError("此帧没有曝光边界")
        else:
            sheet["boundaries"].remove(frame)
        sheet["review_status"] = "revised"
    elif operation == "confirm":
        sheet["review_status"] = "confirmed"
    elif operation == "template":
        template_name = str(params.get("template", ""))
        sheet["phase_template"] = template_name
        sheet["phases"] = formats.phase_template(template_name)
    elif operation == "reset":
        sheet["boundaries"] = list(sheet["estimated_boundaries"])
        sheet["review_status"] = "unreviewed"
        sheet["phases"] = []
        sheet["phase_template"] = ""
    else:
        raise ValueError("未知曝光表操作")
    sheet["author"] = str(params.get("author", sheet["author"]))[:200]
    sheet["updated_at"] = now()
    return commit(session, value, "校正曝光表")


def cut_signature(session, frame):
    if frame not in session.project.boundaries[1:session.project.completed_count]:
        return None
    row = session.project.boundaries.index(frame)
    return [session.project.shots[row - 1].id, session.project.shots[row].id,
            session.project.boundaries[row - 1], frame, session.project.boundaries[row + 1]]


def sound_review(session, frame):
    review = data(session)["sound_reviews"].get(str(frame), {})
    return review if review.get("signature") == cut_signature(session, frame) else {}


def sound_review_put(session, params, emit=None):
    frame = params.get("frame")
    if type(frame) is not int or session.project is None:
        raise ValueError("先选择有效切点")
    signature = cut_signature(session, frame)
    if signature is None:
        raise ValueError("切点已变化，请刷新")
    relation = params.get("relation")
    if relation not in {"J", "L", "sync", "none", "unreviewed"}:
        raise ValueError("声画关系须为 J/L/同步/非声音桥/待核对")
    value = deepcopy(data(session))
    if relation == "unreviewed":
        value["sound_reviews"].pop(str(frame), None)
    else:
        value["sound_reviews"][str(frame)] = dict(signature=signature, relation=relation,
            status="confirmed", author=str(params.get("author", ""))[:200], updated_at=now())
    return commit(session, value, "核对声画关系")


def waveform_run(session, params, cancel=None, progress=None):
    if session.index is None:
        raise ValueError("先打开视频")
    track = session.load_soundtrack(cancel=cancel, progress=progress)
    if cancel is not None and cancel.is_set():
        raise CancelledError("声音观察已取消")
    from shotlab.audio import analyze_audio
    from shotlab.audio import SAMPLE_RATE
    cuts = analyze_audio(session.project, session.index, track)["cuts"] if session.project.completed_count else []
    raw_pcm = getattr(track, "pcm", None)
    spectrum = formats.spectrogram(raw_pcm, SAMPLE_RATE, track.offset_s) if raw_pcm is not None else dict(columns=[])
    segments = formats.sound_segments(track, 0.0, track.duration_s + track.offset_s)
    return dict(project_id=session.project.id, waveform=track.waveform,
                offset_s=track.offset_s, duration_s=track.duration_s,
                cuts=cuts, spectrogram=spectrum, segments=segments,
                note="频谱来自 11 kHz 单声道 PCM 的 NumPy FFT；粗分片段不是 ASR 或音乐分轨")


def _record_put(session, params, collection, builder, label):
    value = deepcopy(data(session))
    rows = value[collection]
    old = item(rows, params["id"]) if params.get("id") else {}
    record = {**old, **builder(session, {**old, **params})}
    record["id"] = old.get("id", uuid4().hex)
    record["created_at"] = old.get("created_at", now())
    record["updated_at"] = now()
    if old:
        rows[rows.index(old)] = record
    else:
        rows.append(record)
    return commit(session, value, label)


def _record_remove(session, params, collection, label):
    value = deepcopy(data(session))
    value[collection].remove(item(value[collection], params["id"]))
    return commit(session, value, label)


def _shot_relation_fields(session, params):
    relation = params.get("relation_type")
    transition = params.get("transition_type", "unknown")
    if relation not in formats.RELATIONS or transition not in formats.TRANSITIONS:
        raise ValueError("选择合法的镜头关系与转场类型")
    source, target = shot_by_id(session, params.get("source_shot_id")), shot_by_id(session, params.get("target_shot_id"))
    frame = params.get("frame")
    if type(frame) is not int or not 0 <= frame < session.project.frame_count:
        raise ValueError("选择有效的证据帧")
    status = params.get("review_status", "confirmed")
    if status not in STATUSES:
        raise ValueError("核对状态无效")
    return dict(relation_type=relation, transition_type=transition,
                source_shot_id=source["id"], target_shot_id=target["id"],
                source_shot_number=source["number"], target_shot_number=target["number"],
                source_start_frame=source["start_frame"], source_end_frame=source["end_frame"],
                target_start_frame=target["start_frame"], target_end_frame=target["end_frame"],
                frame=frame, explanation=formats.text(params.get("explanation"), 3000),
                author=formats.text(params.get("author"), 200), review_status=status,
                source="manual", evidence_ids=list(params.get("evidence_ids") or [])[:20])


def shot_relation_put(session, params, emit=None):
    return _record_put(session, params, "shot_relations", _shot_relation_fields, "编辑镜头关系")


def shot_relation_remove(session, params, emit=None):
    return _record_remove(session, params, "shot_relations", "删除镜头关系")


def relation_report(session, params, emit=None):
    rows = relation_rows(session, data(session)["shot_relations"])
    return dict(project_id=session.project.id, relations=rows, active=sum(r["active"] for r in rows),
                stale=sum(not r["active"] for r in rows))


def relation_rows(session, relations):
    live = {shot.id: (*session.project.interval(i), i + 1)
            for i, shot in enumerate(session.project.shots[:session.project.completed_count])}
    rows = []
    for relation in relations:
        source, target = (live.get(relation[f"{side}_shot_id"]) for side in ("source", "target"))
        active = bool(source and target
                      and source[:2] == (relation["source_start_frame"], relation["source_end_frame"])
                      and target[:2] == (relation["target_start_frame"], relation["target_end_frame"]))
        row = dict(relation, active=active)
        if active:
            row.update(source_shot=source[2], target_shot=target[2])
        rows.append(row)
    return rows


def _evidence_fields(session, params):
    frame = params.get("frame")
    if type(frame) is not int or not 0 <= frame < session.project.frame_count:
        raise ValueError("选择有效证据帧")
    kind = params.get("kind", "documented")
    if kind not in formats.CARD_KINDS:
        raise ValueError("证据卡须标明记录/出处或判断/解释")
    attachment = formats.text(params.get("attachment_path"), 2000)
    attachment_frame = params.get("attachment_frame")
    if attachment:
        if attachment_frame is not None and (type(attachment_frame) is not int or attachment_frame != frame):
            raise ValueError("证据卡帧号与截图帧号不一致")
        path = Path(attachment)
        if not path.is_file():
            raise ValueError("附件路径不存在")
        if path.resolve() in (Path(session.index.path).resolve(),
                              Path(session.project_path).resolve() if session.project_path else None):
            raise ValueError("附件不能是源视频或项目文件")
    _, end = interval(session, dict(start_frame=frame, end_frame=params.get("end_frame", frame + 1)))
    return dict(frame=frame, end_frame=end, attachment_frame=attachment_frame if attachment else None,
                title=formats.text(params.get("title"), 300), kind=kind,
                documented=formats.text(params.get("documented"), 3000),
                interpretation=formats.text(params.get("interpretation"), 3000),
                dialogue=formats.text(params.get("dialogue"), 3000),
                note=formats.text(params.get("note"), 3000),
                bibliography=formats.text(params.get("bibliography"), 2000),
                attachment_path=attachment, attachment_name=Path(attachment).name if attachment else "",
                author=formats.text(params.get("author"), 200), reviewer=formats.text(params.get("reviewer"), 200),
                status=params.get("status", "unreviewed") if params.get("status") in STATUSES else "unreviewed",
                source_version=provenance(session)["source_version"],
                software_version=SOFTWARE_VERSION,
                model=provenance(session)["model"], prompt_version=provenance(session)["prompt_version"])


def evidence_put(session, params, emit=None):
    return _record_put(session, params, "evidence_cards", _evidence_fields, "编辑证据卡")


def evidence_capture(session, params, cancel=None, progress=None):
    from PIL import Image
    from t0.media import FrameReader

    frame = params.get("frame")
    if type(frame) is not int:
        raise ValueError("选择有效证据帧")
    interval(session, dict(start_frame=frame, end_frame=frame + 1))
    if params.get("project_id") != session.project.id:
        raise ValueError("项目已切换，请重新选择证据帧")
    target, _ = formats.safe_output(params["path"], (session.index.path, session.project_path))
    reader = FrameReader(session.index, sequential=False)
    try:
        image = Image.fromarray(reader.read(frame).to_ndarray(format="rgb24"))
    finally:
        reader.close()
    if cancel is not None and cancel.is_set():
        raise CancelledError("证据截图已取消")
    created = False
    try:
        with target.open("xb") as stream:
            created = True
            image.save(stream, format="PNG")
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return dict(project_id=session.project.id, frame=frame, path=str(target))


def evidence_remove(session, params, emit=None):
    return _record_remove(session, params, "evidence_cards", "删除证据卡")


def evidence_report(session, params, emit=None):
    cards = data(session)["evidence_cards"]
    groups = {}
    for card in cards:
        groups.setdefault(str(card["frame"]), []).append(card)
    disagreements = [dict(frame=int(key), cards=rows,
                          reasons=sorted({c["interpretation"] for c in rows if c["interpretation"]}))
                     for key, rows in groups.items()
                     if len({c["interpretation"] for c in rows if c["interpretation"]}) > 1]
    adjudications = data(session)["adjudications"]
    return dict(project_id=session.project.id, cards=cards, disagreements=disagreements,
                adjudications=adjudications,
                note="分歧按证据帧聚合；裁决保留原始意见，不覆盖任何一张卡")


def adjudicate(session, params, emit=None):
    frame = params.get("frame")
    if type(frame) is not int:
        raise ValueError("先选择有分歧的证据帧")
    value = deepcopy(data(session))
    cards = [c for c in value["evidence_cards"] if c["frame"] == frame]
    if not cards:
        raise ValueError("该帧没有证据卡")
    decision = formats.text(params.get("decision"), 3000)
    if not decision.strip():
        raise ValueError("填写裁决结论")
    old = next((a for a in value["adjudications"] if a["frame"] == frame), {})
    record = dict(old, id=old.get("id", uuid4().hex), frame=frame,
                  decision=decision, reason=formats.text(params.get("reason"), 3000),
                  card_ids=[c["id"] for c in cards], author=formats.text(params.get("author"), 200),
                  reviewer=formats.text(params.get("reviewer"), 200),
                  created_at=old.get("created_at", now()), updated_at=now())
    if old:
        value["adjudications"][value["adjudications"].index(old)] = record
    else:
        value["adjudications"].append(record)
    return commit(session, value, "记录证据裁决")


def _speaker_fields(session, params):
    start, end = interval(session, params)
    position = params.get("position", "offscreen")
    if position not in ("onscreen", "offscreen", "unknown"):
        raise ValueError("说话人位置须为画内/画外/待核对")
    return dict(start_frame=start, end_frame=end, speaker=formats.text(params.get("speaker"), 200),
                character=formats.text(params.get("character"), 200),
                shot_id=formats.text(params.get("shot_id"), 100), position=position,
                note=formats.text(params.get("note"), 2000), author=formats.text(params.get("author"), 200),
                status=params.get("status", "unreviewed") if params.get("status") in STATUSES else "unreviewed",
                basis=params.get("basis", "manual"))


def speaker_link_put(session, params, emit=None):
    return _record_put(session, params, "speaker_links", _speaker_fields, "编辑说话人链接")


def speaker_link_remove(session, params, emit=None):
    return _record_remove(session, params, "speaker_links", "删除说话人链接")


def no_dialogue(session, params, emit=None):
    start, end = interval(session, params)
    report = formats.no_dialogue_report(data(session), session, start, end)
    shots = []
    for shot in session.state()["shots"]:
        if shot["pending"] or shot["end"] <= start or shot["start"] >= end:
            continue
        a, b = max(start, shot["start"]), min(end, shot["end"])
        item_report = formats.no_dialogue_report(data(session), session, a, b)
        shots.append(dict(shot_id=shot["id"], shot_number=shot["number"] + 1,
                          start_frame=a, end_frame=b, no_dialogue_s=item_report["no_dialogue_s"],
                          reviewed=item_report["reviewed"]))
    value = data(session)
    scene_tiers = {t["id"] for t in value["tiers"] if t.get("kind") == "scene"}
    scenes = []
    for event in value["events"]:
        if event["tier_id"] not in scene_tiers or event["end_frame"] <= start or event["start_frame"] >= end:
            continue
        a, b = max(start, event["start_frame"]), min(end, event["end_frame"])
        scene_report = formats.no_dialogue_report(value, session, a, b)
        scenes.append(dict(id=event["id"], label=event["label"], start_frame=a, end_frame=b,
                           start_s=endpoint_us(session.index, a) / 1e6,
                           end_s=endpoint_us(session.index, b) / 1e6,
                           duration_s=scene_report["duration_s"],
                           no_dialogue_s=scene_report["no_dialogue_s"],
                           intervals=scene_report["intervals"], reviewed=scene_report["reviewed"],
                           note=scene_report["note"]))
    asr = formats.optional_asr_adapter(params.get("adapter", "none"),
                                       params.get("asr_segments") or params.get("segments"),
                                       params)
    return dict(project_id=session.project.id, range=report, shots=shots, scenes=scenes,
                note="未标注不等于没有对白；场景来自人工 scene 层，声音粗分不代表语音识别", asr=asr)


def _color_fields(session, params):
    start, end = interval(session, params)
    roi = validate_roi(params.get("roi", [0, 0, 1, 1]))
    exclusions = params.get("exclusions", [])
    if not isinstance(exclusions, list) or len(exclusions) > 12:
        raise ValueError("排除区域最多 12 个")
    exclusions = [validate_roi(r) for r in exclusions]
    subject = validate_roi(params["subject"]) if params.get("subject") else None
    observation = params.get("observation") or {}
    if not isinstance(observation, dict):
        raise ValueError("自动观察结果格式错误")
    if observation:
        frame = observation.get("frame")
        if (observation.get("project_id", session.project.id) != session.project.id
                or type(frame) is not int or not start <= frame < end
                or any(observation.get(key) != expected for key, expected in
                       (("roi", roi), ("exclusions", exclusions), ("subject", subject)))):
            raise ValueError("范围或选区已变更，请重新采样")
        observation = {key: observation[key] for key in
                       ("frame", "time_s", "roi", "exclusions", "subject", "color", "composition")
                       if key in observation}
    return dict(start_frame=start, end_frame=end, label=formats.text(params.get("label"), 300),
                scene=formats.text(params.get("scene"), 200), phase=formats.text(params.get("phase"), 100),
                character=formats.text(params.get("character"), 200), roi=roi, exclusions=exclusions,
                subject=subject, gaze=formats.text(params.get("gaze"), 100),
                guides=params.get("guides", ["thirds", "center", "safe_area"]),
                observation=observation,
                judgment=formats.text(params.get("judgment"), 3000),
                author=formats.text(params.get("author"), 200), status="unreviewed")


def color_observation_put(session, params, emit=None):
    return _record_put(session, params, "color_observations", _color_fields, "编辑色彩构图观察")


def color_observation_remove(session, params, emit=None):
    return _record_remove(session, params, "color_observations", "删除色彩构图观察")


def color_analysis_run(session, params, cancel=None, progress=None):
    """Sample a selected source frame and compute ROI/exclusion-mask statistics."""
    import numpy as np
    from PIL import Image
    from t0.media import FrameReader
    start, end = interval(session, params)
    frame = params.get("frame", start)
    if type(frame) is not int or not start <= frame < end:
        raise ValueError("采样帧须位于观察范围内")
    roi = validate_roi(params.get("roi", [0, 0, 1, 1]))
    exclusions = [validate_roi(r) for r in (params.get("exclusions") or [])]
    subject = validate_roi(params["subject"]) if params.get("subject") else None
    if cancel is not None and cancel.is_set():
        raise CancelledError("色彩观察已取消")
    reader = FrameReader(session.index, sequential=False)
    try:
        rgb = reader.read(frame).to_ndarray(format="rgb24")
    finally:
        reader.close()
    if rgb.shape[1] > 480:
        rgb = np.asarray(Image.fromarray(rgb).resize((480, max(2, round(rgb.shape[0] * 480 / rgb.shape[1])))))
    mask = formats.region_mask(rgb.shape[:2], roi, exclusions)
    return dict(project_id=session.project.id, frame=frame, time_s=endpoint_us(session.index, frame) / 1e6,
                roi=roi, exclusions=exclusions,
                subject=subject, color=formats.color_metrics(rgb, mask),
                composition=formats.composition_guides(rgb.shape[:2], subject),
                note="色值来自单帧 sRGB 选区；黑边、字幕带、台标可从排除掩码剔除")


def saved_query_put(session, params, emit=None):
    def fields(_session, merged):
        name = formats.text(merged.get("name"), 200).strip()
        clauses = merged.get("clauses", [])
        if not name or not isinstance(clauses, list) or any(
                not isinstance(c, dict) or c.get("field") not in formats.QUERY_FIELDS for c in clauses):
            raise ValueError("填写查询名称与合法条件")
        formats.validate_query_clauses(clauses)
        return dict(name=name, clauses=clauses[:30], author=formats.text(merged.get("author"), 200),
                    created_at=merged.get("created_at", now()))
    return _record_put(session, params, "saved_queries", fields, "保存组合查询")


def saved_query_remove(session, params, emit=None):
    return _record_remove(session, params, "saved_queries", "删除组合查询")


def query_run(session, params, emit=None):
    clauses = params.get("clauses")
    if clauses is None:
        query = item(data(session)["saved_queries"], params.get("id"))
        clauses = query["clauses"]
    if not isinstance(clauses, list):
        raise ValueError("查询条件格式错误")
    result = formats.query_matches(session.state(), data(session), session.metadata,
                                   (session.motion or {}).get("shots", []), clauses)
    return dict(project_id=session.project.id, **result,
                notes=["保存查询只读取当前项目层；没有运动分析时 stillness 条件不匹配",
                       "跨集查询在 compare.run 中对每个项目分别应用同一组条件"])


def version_alignment_put(session, params, emit=None):
    def fields(_session, merged):
        anchors = merged.get("anchors", [])
        if not isinstance(anchors, list) or not anchors:
            raise ValueError("至少添加一个人工锚点")
        report = formats.manual_alignment(anchors)
        return dict(name=formats.text(merged.get("name"), 200), first_label=formats.text(merged.get("first_label"), 200),
                    second_label=formats.text(merged.get("second_label"), 200),
                    version_note=formats.text(merged.get("version_note"), 1000), anchors=anchors[:300],
                    report=report, author=formats.text(merged.get("author"), 200))
    return _record_put(session, params, "version_alignments", fields, "编辑版本对齐")


def version_alignment_remove(session, params, emit=None):
    return _record_remove(session, params, "version_alignments", "删除版本对齐")


def source_descriptor(index):
    """Identify byte-identical copies without rereading an entire feature film."""
    index.verify_source()
    sample_size = 65536
    digest = hashlib.sha256()
    digest.update(str(index.size).encode())
    with Path(index.path).open("rb") as stream:
        for offset in sorted({0, max(0, (index.size - sample_size) // 2), max(0, index.size - sample_size)}):
            stream.seek(offset)
            digest.update(stream.read(sample_size))
    return dict(path=index.path, size=index.size, mtime_ns=index.mtime_ns, time_base=index.time_base,
                source_origin_pts=index.pts[0], average_rate=index.metadata.get("average_rate"),
                frame_count=len(index.pts), sampled_sha256=digest.hexdigest())


def validate_study_import(session, incoming, source=None, schema_version=None):
    if not isinstance(incoming, dict):
        raise ValueError("文件不含 CadensView study 数据")
    if session.index is None or session.project is None:
        raise ValueError("先打开标注对应的源视频")
    for version in (incoming.get("version", 1), schema_version if schema_version is not None else 1):
        if type(version) is not int or not 1 <= version <= STUDY_VERSION:
            raise ValueError("不支持的研究标注版本")
    if source is not None:
        actual = source_descriptor(session.index)
        keys = ("size", "time_base", "source_origin_pts", "frame_count", "sampled_sha256")
        if not isinstance(source, dict) or any(k in source and source[k] != actual[k] for k in keys):
            raise ValueError("标注包与当前源视频不匹配")
        if ("sampled_sha256" not in source and source.get("mtime_ns") is not None
                and source.get("mtime_ns") != actual["mtime_ns"]):
            raise ValueError("旧标注包的源文件版本不匹配，请打开原始视频")
    if type(incoming.get("subtitles_imported", False)) is not bool or not isinstance(incoming.get("provenance", {}), dict):
        raise ValueError("标注包的来源信息格式错误")

    def fail(key):
        raise ValueError(f"标注包的 {key} 记录无效")

    def strings(row, names, key):
        if any(not isinstance(row.get(name), str) for name in names):
            fail(key)

    tiers = {t["id"] for t in data(session)["tiers"]}
    for key, default in empty().items():
        if not isinstance(default, list):
            continue
        rows = incoming.get(key, [])
        if not isinstance(rows, list):
            fail(key)
        ids = set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
                fail(key)
            ids.add(row["id"])
            if key == "tiers":
                strings(row, ("name", "kind"), key)
                if (row["kind"] not in KINDS or type(row.get("version")) is not int or row["version"] < 1
                        or not isinstance(row.get("vocabulary"), list)
                        or any(not isinstance(word, str) for word in row["vocabulary"])):
                    fail(key)
                tiers.add(row["id"])
            if key in ("events", "exposures", "speaker_links", "color_observations"):
                interval(session, row)
            if key == "events":
                strings(row, ("tier_id", "label", "note", "speaker", "author", "source", "review_status"), key)
                if row["tier_id"] not in tiers or not row["label"].strip() or row["review_status"] not in STATUSES:
                    fail(key)
            if key == "exposures":
                strings(row, ("name", "review_status", "author"), key)
                validate_roi(row.get("roi"))
                for field in ("boundaries", "estimated_boundaries"):
                    b = row.get(field)
                    if (not isinstance(b, list) or len(b) < 2 or any(type(f) is not int for f in b)
                            or b[0] != row["start_frame"] or b[-1] != row["end_frame"]
                            or any(a >= z for a, z in zip(b, b[1:]))):
                        fail(key)
                if (not isinstance(row.get("phases", []), list)
                        or any(not isinstance(p, dict) for p in row.get("phases", []))
                        or not isinstance(row.get("changes", []), list)
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in row.get("changes", []))
                        or (row.get("warning") is not None and not isinstance(row["warning"], str))
                        or not isinstance(row.get("phase_template", ""), str)):
                    fail(key)
            if key in ("shot_relations", "evidence_cards", "adjudications"):
                frame = row.get("frame")
                if type(frame) is not int or not 0 <= frame < session.project.frame_count:
                    fail(key)
            if key == "shot_relations":
                strings(row, ("relation_type", "source_shot_id", "target_shot_id"), key)
                if row["relation_type"] not in formats.RELATIONS:
                    fail(key)
                for side in ("source", "target"):
                    interval(session, {f: row.get(f"{side}_{f}") for f in ("start_frame", "end_frame")})
            if key == "evidence_cards":
                strings(row, ("interpretation",), key)
                interval(session, dict(start_frame=row["frame"], end_frame=row.get("end_frame", row["frame"] + 1)))
            if key == "saved_queries":
                strings(row, ("name",), key)
                formats.validate_query_clauses(row.get("clauses"))
            if key == "version_alignments":
                formats.manual_alignment(row.get("anchors"))
    reviews = incoming.get("sound_reviews", {})
    if not isinstance(reviews, dict):
        fail("sound_reviews")
    for key, review in reviews.items():
        if (not key.isascii() or not key.isdecimal() or not 0 <= int(key) < session.project.frame_count
                or not isinstance(review, dict) or not isinstance(review.get("signature"), list)
                or not isinstance(review.get("relation"), str)):
            fail("sound_reviews")


def merge_study(session, incoming, params, *, format_name, source=None, schema_version=None, count_tiers=True):
    validate_study_import(session, incoming, source, schema_version)
    try:
        json.dumps(incoming, allow_nan=False)
    except ValueError as error:
        raise ValueError("标注包包含无效数值") from error
    original = data(session)
    value = deepcopy(original)
    imported = conflicts = 0
    for key, rows in value.items():
        if not isinstance(rows, list):
            continue
        known = {r["id"]: r for r in rows}
        for row in incoming.get(key, []):
            if row["id"] not in known:
                rows.append(row)
                imported += int(key != "tiers" or count_tiers)
            elif key == "events" and any(row[f] != known[row["id"]][f]
                                         for f in ("label", "note", "start_frame", "end_frame")):
                conflicts += 1
    for frame, review in incoming.get("sound_reviews", {}).items():
        if frame not in value["sound_reviews"]:
            value["sound_reviews"][frame] = review
            imported += 1
    value["subtitles_imported"] |= incoming.get("subtitles_imported", False)
    if params.get("provenance"):
        value["provenance"] = dict(value["provenance"], bibliography=str(params["provenance"])[:1000])
    state = commit(session, value, "导入标注") if value != original else session.state()
    state["study_import"] = dict(imported=imported, conflicts=conflicts, format=format_name)
    return state


def import_bundle(session, params, emit=None):
    try:
        bundle = json.loads(Path(params["path"]).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise ValueError("无法读取标注 JSON") from error
    if not isinstance(bundle, dict):
        raise ValueError("文件不含 CadensView study 数据")
    return merge_study(session, bundle.get("study"), params, format_name="CadensView JSON",
                       source=bundle.get("source"), schema_version=bundle.get("schema_version"))


def import_eaf(session, params, emit=None):
    payload = Path(params["path"])
    if not payload.is_file():
        raise ValueError("EAF 文件不存在")
    if session.index is None or session.project is None:
        raise ValueError("先打开标注对应的源视频")
    parsed, kind = formats.parse_eaf(payload.read_bytes())
    index = session.index
    pts = [endpoint_us(index, f) for f in range(len(index.pts) + 1)]

    def frame_at(ms, endpoint):
        us = ms * 1000
        after = bisect_left(pts, us)
        nearby = range(max(0, after - 1), min(len(pts), after + 1))
        nearest = min(nearby, key=lambda f: abs(pts[f] - us))
        if abs(pts[nearest] - us) <= 500:
            return nearest
        return min(len(pts) - 1, after) if endpoint else max(0, bisect_right(pts, us) - 1)

    if kind == "bundle":
        source = parsed.pop("source", None)
        validate_study_import(session, parsed, source)
        formats.apply_eaf_event_edits(parsed, pts, frame_at)
        return merge_study(session, parsed, params, format_name="EAF 3.0 + CadensView projection")
    events = []
    for event in parsed["events"]:
        a, b = event.pop("milliseconds")
        if a < 0 or b <= a or a * 1000 >= pts[-1]:
            continue
        start, end = frame_at(a, False), frame_at(b, True)
        if start < end:
            events.append(dict(event, start_frame=start, end_frame=end, template_version=1, updated_at=now()))
    parsed["events"] = events
    return merge_study(session, parsed, params, count_tiers=False,
                       format_name="EAF 3.0 generic tiers; milliseconds mapped to source frames")


def validate_reference(session, params, emit=None):
    references = params.get("references")
    predictions = params.get("predictions")
    if not isinstance(references, dict) or not isinstance(predictions, dict):
        raise ValueError("参考集与预测集须为按视频分组的帧区间")
    result = formats.clipshots_metrics(references, predictions, params.get("tolerance", 0))
    result.update(source=formats.text(params.get("source"), 500), reference_version=formats.text(params.get("version"), 200),
                  note=result["note"] + "；仅供人工核对，未运行真实素材验收")
    return dict(project_id=session.project.id, **result)


def asr_status(session, params, emit=None):
    result = formats.optional_asr_adapter(params.get("adapter", "none"),
                                         params.get("segments") or params.get("asr_segments"), params)
    return dict(project_id=session.project.id, **result)


def difference_run(session, params, cancel=None, progress=None):
    """Export a difference image and continuous numbered frames from a corrected sheet."""
    import numpy as np
    from PIL import Image
    from t0.media import FrameReader
    sheet = item(data(session)["exposures"], params.get("sheet_id"))
    start, end = int(sheet["start_frame"]), int(sheet["end_frame"])
    if end - start > MAX_EXPOSURE_FRAMES:
        raise ValueError(f"连续帧导出最多 {MAX_EXPOSURE_FRAMES} 帧")
    target = Path(params["path"]).resolve()
    protected = [Path(session.index.path).resolve()]
    if session.project_path:
        protected.append(Path(session.project_path).resolve())
    frames_dir = target.parent / (target.stem + "_frames")
    difference_path = target.parent / (target.stem + "_difference.png")
    manifest_path = target.parent / (target.stem + "_manifest.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    outputs = [frames_dir, difference_path, manifest_path]
    for output in outputs:
        for source in protected:
            if output.resolve() == source or (output.exists() and source.exists() and output.samefile(source)):
                raise ValueError("不能覆盖源视频或项目文件")
    if any(p.exists() for p in outputs):
        raise ValueError("导出目录或差图已存在，请使用新的名称")
    roi = validate_roi(sheet["roi"])
    reader = FrameReader(session.index, sequential=True)
    created = []
    try:
        frames_dir.mkdir(parents=True, exist_ok=False)
        created.append(frames_dir)
        previous = None
        difference_total = None
        entries = []
        for frame in range(start, end):
            if cancel is not None and cancel.is_set():
                raise CancelledError("连续帧导出已取消")
            rgb = reader.read(frame).to_ndarray(format="rgb24")
            height, width = rgb.shape[:2]
            x, y, w, h = roi
            cropped = rgb[int(y * height):min(height, math.ceil((y + h) * height)),
                          int(x * width):min(width, math.ceil((x + w) * width))]
            if previous is not None and cropped.shape == previous.shape:
                difference = np.abs(cropped.astype(np.int16) - previous.astype(np.int16)).mean(axis=2)
                scaled = np.clip(difference * 3, 0, 255).astype(np.uint8)
                difference_total = scaled.astype(np.float64) if difference_total is None \
                    else difference_total + scaled.astype(np.float64)
            previous = cropped
            name = f"{frame - start + 1:06d}.png"
            Image.fromarray(cropped).save(frames_dir / name)
            entries.append(dict(number=frame - start + 1, file=name, source_frame=frame,
                                start_s=endpoint_us(session.index, frame) / 1e6,
                                end_s=endpoint_us(session.index, frame + 1) / 1e6,
                                review_status=sheet["review_status"], roi=roi))
            if progress is not None:
                progress(frame - start + 1, end - start)
        if cancel is not None and cancel.is_set():
            raise CancelledError("连续帧导出已取消")
        mean = difference_total / max(1, len(entries) - 1) if difference_total is not None \
            else np.zeros_like(previous.mean(axis=2), dtype=np.float64)
        with difference_path.open("xb") as stream:
            created.append(difference_path)
            Image.fromarray(np.rint(mean).astype(np.uint8), "L").save(stream, format="PNG")
        manifest = dict(sheet_id=sheet["id"], source=dict(path=session.index.path, mtime_ns=session.index.mtime_ns),
                        provenance=provenance(session), interval="source display frame [start,end)",
                        boundaries=sheet["boundaries"], estimated_boundaries=sheet["estimated_boundaries"],
                        phases=sheet.get("phases", []), frames=entries,
                        note="编号按修订后区间连续写出；差图是相邻帧绝对差的平均值，不是原画张数")
        with manifest_path.open("x", encoding="utf-8") as stream:
            created.append(manifest_path)
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
    except BaseException:
        for path in reversed(created):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        raise
    finally:
        reader.close()
    return dict(project_id=session.project.id, path=str(manifest_path), frames=str(frames_dir),
                difference=str(difference_path), written=end - start)


def export(session, params, emit=None):
    """JSON bundle, EAF projection and CSVs, containing source coordinates."""
    if session.index is None:
        raise ValueError("先打开视频")
    target = Path(params["path"]).resolve().with_suffix(".json")
    events_path = target.with_suffix(".events.csv")
    exposures_path = target.with_suffix(".exposures.csv")
    eaf_path = target.with_suffix(".eaf")
    relations_path = target.with_suffix(".relations.csv")
    cards_path = target.with_suffix(".evidence.csv")
    protected = [Path(session.index.path).resolve()]
    if session.project_path:
        protected.append(Path(session.project_path).resolve())
    outputs = [target, events_path, exposures_path, eaf_path, relations_path, cards_path]
    target, outputs = formats.safe_output(target, protected, outputs[1:])
    value = data(session)
    index = session.index
    bundle = dict(software="CadensView", schema_version=STUDY_VERSION, exported_at=now(),
                  provenance=provenance(session),
                  source=source_descriptor(index),
                  interval="source display frame [start, end); seconds relative to first frame",
                  metadata=session.metadata, study=value,
                  notes=["曝光为成片可见变化估计，不等于制作原画张数", "声画候选必须与人工核对结果分开使用",
                         "EAF 是公开 3.0 格式的受限投影；完整字段以本 JSON 为准"])
    tiers = {t["id"]: t for t in value["tiers"]}
    rows = []
    for e in value["events"]:
        rows.append(dict(e, tier_name=tiers[e["tier_id"]]["name"], kind=tiers[e["tier_id"]]["kind"],
                         start_s=endpoint_us(index, e["start_frame"]) / 1e6,
                         end_s=endpoint_us(index, e["end_frame"]) / 1e6))
    exposures = []
    for sheet in value["exposures"]:
        phases = {str(p.get("index")): p.get("label", "") for p in sheet.get("phases", [])}
        for phase_number, (a, b) in enumerate(zip(sheet["boundaries"], sheet["boundaries"][1:]), 1):
            exposures.append(dict(sheet_id=sheet["id"], name=sheet["name"], start_frame=a, end_frame=b,
                                  held_frames=b - a, roi=json.dumps(sheet["roi"]),
                                  phase_template=sheet.get("phase_template", ""), phase=phases.get(str(phase_number), ""),
                                  review_status=sheet["review_status"], author=sheet["author"]))
    relations = relation_rows(session, value["shot_relations"])
    for row in relations:
        row["relation_label"] = formats.RELATIONS[row["relation_type"]]
    cards = value["evidence_cards"]
    target.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with target.open("x", encoding="utf-8") as stream:
            created.append(target)
            json.dump(bundle, stream, ensure_ascii=False, indent=2)
        with eaf_path.open("x", encoding="utf-8") as stream:
            created.append(eaf_path)
            stream.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                         + formats.eaf_xml(session, value, source=bundle["source"]) + "\n")
        for path, entries, default in [
            (events_path, rows, ["id", "tier_id", "label", "start_frame", "end_frame"]),
            (exposures_path, exposures, ["sheet_id", "start_frame", "end_frame", "held_frames"]),
            (relations_path, relations, ["id", "relation_type", "source_shot_id", "target_shot_id", "explanation"]),
            (cards_path, cards, ["id", "frame", "title", "kind", "author", "status"]),
        ]:
            fields = formats.csv_rows(entries, default)
            with path.open("x", encoding="utf-8-sig", newline="") as stream:
                created.append(path)
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(entries)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return dict(path=str(target), events=str(events_path), exposures=str(exposures_path),
                eaf=str(eaf_path), relations=str(relations_path), evidence=str(cards_path))


def register(methods, jobs):
    methods.update({"study.template": template, "study.tier.put": tier_put, "study.tier.remove": tier_remove,
                    "study.event.put": event_put, "study.event.remove": event_remove,
                    "study.import_subtitles": import_subtitles, "study.exposure.edit": exposure_edit,
                    "study.sound_review": sound_review_put, "study.export": export,
                    "study.relation.put": shot_relation_put, "study.relation.remove": shot_relation_remove,
                    "study.relation.report": relation_report,
                    "study.evidence.put": evidence_put, "study.evidence.remove": evidence_remove,
                    "study.evidence.report": evidence_report, "study.evidence.adjudicate": adjudicate,
                    "study.speaker.put": speaker_link_put, "study.speaker.remove": speaker_link_remove,
                    "study.no_dialogue": no_dialogue, "study.asr_status": asr_status,
                    "study.color.put": color_observation_put, "study.color.remove": color_observation_remove,
                    "study.query.put": saved_query_put, "study.query.remove": saved_query_remove,
                    "study.query.run": query_run, "study.alignment.put": version_alignment_put,
                    "study.alignment.remove": version_alignment_remove,
                    "study.import_json": import_bundle, "study.import_eaf": import_eaf,
                    "study.validate_reference": validate_reference})
    jobs.update({"study.exposure.run": ("study", exposure_run), "study.waveform.run": ("study", waveform_run),
                 "study.evidence.capture": ("study", evidence_capture),
                 "study.color.run": ("study", color_analysis_run), "study.difference.run": ("study", difference_run)})
