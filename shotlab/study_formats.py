"""Interchange and reusable analysis primitives for the research workspace.

EAF 3.0 is a lossy projection of the richer CadensView JSON study.  The XML is
generated from the public MPI schema, not copied from ELAN source.  Complex
records are preserved as JSON in HEADER/PROPERTY so a round trip can restore
authored, reviewed and versioned fields exactly.
"""

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

RELATIONS = {
    "shot_reverse": "正打/反打", "reaction": "反应镜头", "eyeline": "视线匹配",
    "action_match": "动作接戏", "graphic_match": "图形匹配", "insert": "插入镜头",
    "parallel": "平行剪辑", "motif": "重复母题", "transition": "转场", "other": "其他",
}
TRANSITIONS = {"cut", "dissolve", "fade_in", "fade_out", "wipe", "iris", "match_cut", "other", "unknown"}
PHASES = {"anticipation": "预备", "action": "动作", "buffer": "缓冲", "settle": "收势", "hold": "保持"}
CARD_KINDS = {"documented": "记录 / 出处", "interpretation": "判断 / 解释"}
QUERY_FIELDS = {"text", "tag", "scale", "camera", "duration_min", "duration_max",
                "stillness_min", "stillness_max", "metadata", "kind", "all", "any", "not"}
ALIGNMENT_KINDS = {"composition_similarity", "material_reuse"}
EAF_KIND = "cadensview-study-kind"
EAF_VERSION = "cadensview-study-schema"
EAF_BUNDLE = "cadensview-json"
EAF_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")


def text(value, limit=5000):
    return str(value if value is not None else "")[:limit]


def identity(value):
    value = str(value)
    return value if EAF_ID.match(value) else "cv_" + hashlib.sha1(value.encode()).hexdigest()[:20]


def safe_output(target, protected=(), siblings=()):
    target = Path(target).resolve()
    protected = [Path(p).resolve() for p in protected if p]
    outputs = [target, *(Path(p).resolve() for p in siblings)]
    for output in outputs:
        for source in protected:
            if output == source or (output.exists() and source.exists() and output.samefile(source)):
                raise ValueError("不能覆盖源视频或项目文件")
    if any(p.exists() for p in outputs):
        raise ValueError("导出文件已存在，请使用新的名称")
    return target, outputs


def csv_rows(entries, default):
    fields = list(dict.fromkeys(k for row in entries for k in row)) or list(default)
    return fields


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _element(parent, name, attributes=None, value=None):
    node = ET.SubElement(parent, name, {k: str(v) for k, v in (attributes or {}).items()})
    if value is not None:
        node.text = str(value)
    return node


def _alignable_tiers(value):
    """(tier id, tier label, records) for every time-aligned collection."""
    tiers = {t["id"]: t["name"] for t in value.get("tiers", [])}
    entries = []
    for tier in value.get("tiers", []):
        rows = [e for e in value.get("events", []) if e.get("tier_id") == tier["id"]]
        if rows:
            entries.append(("tier_" + tier["id"], text(tier["name"], 120), tier["kind"], rows))
    for sheet in value.get("exposures", []):
        rows = []
        for number, (a, b) in enumerate(zip(sheet["boundaries"], sheet["boundaries"][1:]), 1):
            rows.append(dict(id=f"{sheet['id']}-{number}", start_frame=a, end_frame=b,
                             label=f"{sheet['name']} · 第{number}段", note=text(sheet.get("phase_template", ""))))
        entries.append(("exposure_" + sheet["id"], "曝光 · " + text(sheet["name"], 100), "exposure", rows))
    for card in value.get("evidence_cards", []):
        frame = int(card["frame"])
        entries.append(("evidence_" + card["id"], "证据 · " + text(card.get("title", ""), 100), "evidence",
                        [dict(id=card["id"], start_frame=frame, end_frame=frame + 1,
                              label=text(card.get("title"), 500), note=text(card.get("note"), 1000))]))
    for observation in value.get("color_observations", []):
        start, end = int(observation["start_frame"]), int(observation["end_frame"])
        entries.append(("color_" + observation["id"], "色彩/构图 · " + text(observation.get("scene", ""), 100),
                        "color", [dict(id=observation["id"], start_frame=start, end_frame=end,
                                       label=text(observation.get("label"), 500), note=text(observation.get("judgment"), 1000))]))
    relations = {}
    for relation in value.get("shot_relations", []):
        relations.setdefault(relation["relation_type"], []).append(
            [dict(id=f"{relation['id']}-{side}", start_frame=relation[f"{side}_start_frame"],
                  end_frame=max(relation[f"{side}_start_frame"] + 1, relation[f"{side}_end_frame"]),
                  label=text(RELATIONS.get(relation["relation_type"], relation["relation_type"]), 100),
                  note=text(relation.get("explanation"), 1000)) for side in ("source", "target")])
    for kind, pairs in relations.items():
        rows = [row for pair in pairs for row in pair]
        entries.append(("relation_" + identity(kind), "镜头关系 · " + text(RELATIONS.get(kind, kind), 100),
                        "relation", rows))
    for link in value.get("speaker_links", []):
        entries.append(("speaker_" + link["id"], "声音链接 · " + text(link.get("speaker", ""), 100), "speaker",
                        [dict(id=link["id"], start_frame=int(link["start_frame"]), end_frame=int(link["end_frame"]),
                              label=text(link.get("speaker"), 200), note=text(link.get("note"), 1000))]))
    return tiers, entries


def eaf_xml(session, value, source=None):
    from shotlab.study import endpoint_us
    import mimetypes
    index = session.index
    root = ET.Element("ANNOTATION_DOCUMENT", {
        "AUTHOR": "CadensView", "DATE": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "VERSION": "3.0", "FORMAT": "3.0",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://www.mpi.nl/tools/elan/EAFv3.0.xsd"})
    rights = (value.get("provenance") or {})
    _element(root, "LICENSE", None,
             "标注数据权利由数据作者声明；源媒体权利单独处理。"
             + (f" 作者：{text(rights.get('author'), 200)}；" if rights.get("author") else "")
             + (f" 出处：{text(rights.get('bibliography'), 500)}；" if rights.get("bibliography") else "")
             + "本文件不自动授予任何版权许可。")
    header = _element(root, "HEADER", {"TIME_UNITS": "milliseconds"})
    mime = mimetypes.guess_type(index.path)[0] or "application/octet-stream"
    _element(header, "MEDIA_DESCRIPTOR", {"MEDIA_URL": Path(index.path).as_uri(), "MIME_TYPE": mime})
    _element(header, "PROPERTY", {"NAME": EAF_VERSION}, str(value.get("version", 1)))
    _, collections = _alignable_tiers(value)
    for tier_id, _, kind, _ in collections:
        _element(header, "PROPERTY", {"NAME": f"{EAF_KIND}:{identity(tier_id)}"}, kind)
    _element(header, "PROPERTY", {"NAME": EAF_BUNDLE},
             _json(dict(version=value.get("version", 1), source=source, provenance=rights, tiers=value.get("tiers", []),
                        events=value.get("events", []), exposures=value.get("exposures", []),
                        sound_reviews=value.get("sound_reviews", {}), shot_relations=value.get("shot_relations", []),
                        evidence_cards=value.get("evidence_cards", []), adjudications=value.get("adjudications", []),
                        speaker_links=value.get("speaker_links", []), color_observations=value.get("color_observations", []),
                        version_alignments=value.get("version_alignments", []), saved_queries=value.get("saved_queries", []),
                        subtitles_imported=bool(value.get("subtitles_imported")))))
    time_order = _element(root, "TIME_ORDER")
    slots = {}

    def slot(frame):
        milliseconds = max(0, round(endpoint_us(index, int(frame)) / 1000))
        if milliseconds not in slots:
            key = f"ts{len(slots) + 1}"
            _element(time_order, "TIME_SLOT", {"TIME_SLOT_ID": key, "TIME_VALUE": milliseconds})
            slots[milliseconds] = key
        return slots[milliseconds]

    annotation_refs = {}
    for index_number, (tier_id, label, kind, rows) in enumerate(collections, 1):
        tier = _element(root, "TIER", {"TIER_ID": identity(tier_id), "LINGUISTIC_TYPE_REF": "cv-alignable",
                                       "PARTICIPANT": label})
        for number, row in enumerate(rows, 1):
            annotation = _element(tier, "ANNOTATION")
            anno_id = identity(f"a{index_number}_{number}_{row['id']}")
            alignable = _element(annotation, "ALIGNABLE_ANNOTATION", {"ANNOTATION_ID": anno_id,
                                 "TIME_SLOT_REF1": slot(row["start_frame"]), "TIME_SLOT_REF2": slot(row["end_frame"])})
            _element(alignable, "ANNOTATION_VALUE", None, eaf_annotation_text(row))
            annotation_refs[f"cv:{tier_id}:{row['id']}"] = anno_id
    # Even an unconstrained tier obtains the small built-in vocabulary of
    # relationship/phase terms; the complete, versioned word list remains in
    # the JSON projection in HEADER/PROPERTY.
    vocabulary = list(dict.fromkeys(
        [str(word).strip() for tier in value.get("tiers", [])
         for word in tier.get("vocabulary", []) if str(word).strip()]
        + list(RELATIONS.values()) + list(PHASES.values())))
    cv_id = "cv-study-vocabulary"
    _element(root, "LINGUISTIC_TYPE", {"LINGUISTIC_TYPE_ID": "cv-alignable", "TIME_ALIGNABLE": "true",
                                       "CONTROLLED_VOCABULARY_REF": cv_id})
    _element(root, "LANGUAGE", {"LANG_ID": "cv-lang", "LANG_LABEL": "CadensView study terms"})
    controlled = _element(root, "CONTROLLED_VOCABULARY", {"CV_ID": cv_id})
    for number, word in enumerate(vocabulary, 1):
        entry = _element(controlled, "CV_ENTRY_ML", {"CVE_ID": identity(f"cve_{number}_{word}")})
        _element(entry, "CVE_VALUE", {"LANG_REF": "cv-lang"}, word)
    relations = [r for r in value.get("shot_relations", []) if r.get("source_shot_id") and r.get("target_shot_id")]
    if relations:
        link_set = _element(root, "REF_LINK_SET", {"LINK_SET_ID": "cv-shot-relations",
                                                   "LINK_SET_NAME": "CadensView typed shot relationships"})
        for number, relation in enumerate(relations, 1):
            left = f"cv:relation_{identity(relation['relation_type'])}:{relation['id']}-source"
            right = f"cv:relation_{identity(relation['relation_type'])}:{relation['id']}-target"
            if left not in annotation_refs or right not in annotation_refs:
                continue
            _element(link_set, "CROSS_REF_LINK", {
                "REF_LINK_ID": identity(f"r{number}_{relation['id']}"),
                "REF_LINK_NAME": text(relation.get("explanation"), 200),
                "REF_TYPE": relation["relation_type"], "REF1": annotation_refs[left],
                "REF2": annotation_refs[right], "DIRECTIONALITY": "unidirectional"},
                RELATIONS.get(relation["relation_type"], relation["relation_type"]))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=False)


def eaf_annotation_text(row):
    return f"{text(row.get('label'), 500)}\n{text(row.get('note'), 1000)}"


def eaf_value(annotation):
    value = annotation.find("ANNOTATION_VALUE")
    return ((value.text if value is not None else annotation.text) or "").replace("\r\n", "\n")


def apply_eaf_event_edits(bundle, pts, frame_at):
    """Overlay XML edits on timed study events; keep rich, unedited JSON fields."""
    xml_rows = bundle.pop("_eaf_annotations")
    _, collections = _alignable_tiers(bundle)
    event_tiers = {"tier_" + tier["id"] for tier in bundle.get("tiers", [])}
    for tier_number, (tier, _, _, rows) in enumerate(collections, 1):
        if tier not in event_tiers:
            continue
        for number, row in enumerate(rows, 1):
            edit = xml_rows.get((identity(tier), identity(f"a{tier_number}_{number}_{row['id']}")))
            if edit is None:  # XML deletions do not silently remove stored annotations.
                continue
            full, times = edit
            changed = False
            if full.strip() != eaf_annotation_text(row).strip() and full.strip():
                previous_note = "\n" + text(row.get("note"), 1000)
                label = full.removesuffix(previous_note).strip()
                if label:
                    row["label"] = label[:5000]
                    changed = True
            for i, field in enumerate(("start_frame", "end_frame")):
                ms = times[i]
                if ms is not None and ms != round(pts[row[field]] / 1000):
                    if ms < 0 or ms * 1000 > pts[-1] + 500:
                        raise ValueError("EAF 修改后的时间超出当前源视频")
                    row[field] = frame_at(ms, i == 1)
                    changed = True
            if changed:
                row.update(source="eaf", review_status="unreviewed",
                           updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def parse_eaf(payload):
    """Return the embedded CadensView projection, or alignable tiers as new events."""
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as error:
        raise ValueError("无法读取 EAF 标注文档") from error
    if root.tag != "ANNOTATION_DOCUMENT":
        raise ValueError("不是 EAF 标注文档")
    header = root.find("HEADER")
    if header is not None and header.get("TIME_UNITS", "milliseconds") != "milliseconds":
        raise ValueError("仅支持使用毫秒时间槽的 EAF")
    properties = {p.get("NAME"): p.text or "" for p in (header.findall("PROPERTY") if header is not None else [])}
    time_slots = {slot.get("TIME_SLOT_ID"): int(slot.get("TIME_VALUE"))
                  for slot in root.findall("TIME_ORDER/TIME_SLOT") if slot.get("TIME_VALUE") is not None}
    if EAF_BUNDLE in properties:
        try:
            bundle = json.loads(properties[EAF_BUNDLE])
        except ValueError as error:
            raise ValueError("EAF 内的 CadensView 数据已损坏") from error
        if not isinstance(bundle, dict):
            raise ValueError("EAF 内的 CadensView 数据格式错误")
        bundle["_eaf_annotations"] = {
            (tier.get("TIER_ID"), annotation.get("ANNOTATION_ID")): (
                eaf_value(annotation), [time_slots.get(annotation.get(key))
                                        for key in ("TIME_SLOT_REF1", "TIME_SLOT_REF2")])
            for tier in root.findall("TIER")
            for annotation in tier.findall("ANNOTATION/ALIGNABLE_ANNOTATION")
        }
        return bundle, "bundle"
    known_kinds = {"scene", "character", "dialogue", "music", "camera", "action", "custom"}
    vocabulary = [node.text or "" for node in root.findall("CONTROLLED_VOCABULARY/CV_ENTRY_ML/CVE_VALUE")]
    tiers, events = [], []
    for tier in root.findall("TIER"):
        tier_id = tier.get("TIER_ID") or ""
        name = tier.get("PARTICIPANT") or tier_id
        kind = properties.get(f"{EAF_KIND}:{identity(tier_id)}", "")
        if kind not in known_kinds:
            kind = "dialogue" if "对白" in name else "custom"
        record = dict(id=tier_id, name=name, kind=kind,
                      vocabulary=[v.strip() for v in vocabulary if v.strip()][:200],
                      definition="EAF 3.0 导入", version=1)
        tiers.append(record)
        for annotation in tier.findall("ANNOTATION/ALIGNABLE_ANNOTATION"):
            first, last = annotation.get("TIME_SLOT_REF1"), annotation.get("TIME_SLOT_REF2")
            if first not in time_slots or last not in time_slots:
                continue
            full = eaf_value(annotation).strip()
            label = full.splitlines()[0] if full else ""
            if not label:
                continue
            note = "\n".join(full.splitlines()[1:]) or "EAF 毫秒时间槽导入"
            events.append(dict(id=identity(f"{tier_id}-{annotation.get('ANNOTATION_ID')}"), tier_id=tier_id,
                               label=label[:5000], speaker="", note=note[:5000], author="",
                               source="eaf", review_status="unreviewed", milliseconds=[time_slots[first], time_slots[last]],
                               created_at=datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return dict(tiers=tiers, events=events), "generic"


def clipshots_metrics(references, predictions, tolerance=0):
    """Overlap-based interval precision/recall/F1 for cuts and gradual transitions.

    `references`/`predictions` map video names to lists of [begin, end) frame
    intervals. A one-frame interval is a cut; a longer one is gradual. This is
    an independent implementation of the ClipShots evaluation idea.
    """
    if type(tolerance) is not int or tolerance < 0:
        raise ValueError("帧容差必须是非负整数")

    def validate(collection):
        for rows in collection.values():
            if not isinstance(rows, list) or any(
                not isinstance(r, (list, tuple)) or len(r) != 2 or any(type(v) is not int for v in r)
                or not 0 <= r[0] < r[1] for r in rows
            ):
                raise ValueError("参考与预测区间须为非空半开帧区间 [begin,end)")

    validate(references)
    validate(predictions)

    def match_count(left, right):
        # Earliest finish first. Of its compatible partners take the earliest
        # finish: exchanging two such pairs preserves overlap and cardinality.
        left = sorted(left, key=lambda r: r[1])
        right = sorted(((a - tolerance, b + tolerance) for a, b in right), key=lambda r: r[1])
        groups = (left, right)
        used = (set(), set())
        edges = sorted((b, group, i) for group, rows in enumerate(groups) for i, (_, b) in enumerate(rows))
        count = 0
        for end, group, i in edges:
            if i in used[group]:
                continue
            start = groups[group][i][0]
            other = 1 - group
            for j, (a, b) in enumerate(groups[other]):
                if j not in used[other] and a < end and start < b:
                    used[other].add(j)
                    count += 1
                    break
            used[group].add(i)
        return count

    def partition(rows):
        return ([r for r in rows if r[1] - r[0] <= 1], [r for r in rows if r[1] - r[0] > 1])

    result = {}
    for label, ref_index, pred_index in (("cut", 0, 0), ("gradual", 1, 1), ("all", None, None)):
        matches = refs = preds = 0
        for name in references.keys() | predictions.keys():
            intervals = references.get(name, [])
            reference, prediction = partition(intervals), partition(predictions.get(name, []))
            left = list(intervals if label == "all" else reference[ref_index])
            right = list(predictions.get(name, []) if label == "all" else prediction[pred_index])
            refs += len(left)
            preds += len(right)
            matches += match_count(left, right)
        precision = matches / preds if preds else 0.0
        recall = matches / refs if refs else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = dict(matched=matches, reference=refs, predicted=preds,
                             precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4))
    result["tolerance_frames"] = tolerance
    result["note"] = "One-to-one matches of overlapping [begin,end) intervals; a one-frame interval is a hard cut"
    return result


def spectrogram(pcm, sample_rate, offset_s=0.0, window=1024, hop=256, bands=32, max_columns=1200):
    """Compact log-frequency spectrogram bins for the GUI (no plotting dependency)."""
    import numpy as np
    values = np.asarray(pcm, dtype=np.float32).ravel()
    if not values.size or window < 8 or hop < 1:
        return dict(offset_s=round(float(offset_s), 5), hop_s=0.0, bands=0, columns=[])
    count = (values.size - window) // hop + 1 if values.size >= window else 1
    padded = values if values.size >= window else np.pad(values, (0, window - values.size))
    edges = np.geomspace(60, max(61.0, sample_rate / 2), bands + 1)
    bins = np.fft.rfftfreq(window, 1 / sample_rate)
    band_of = np.clip(np.searchsorted(edges, bins, side="right") - 1, 0, bands - 1)
    onehot = np.zeros((bins.size, bands), dtype=np.float32)
    onehot[np.arange(bins.size), band_of] = 1.0
    taper = np.hanning(window).astype(np.float32)
    stride = max(1, math.ceil(count / max_columns))
    columns = []
    for start in range(0, count, stride):
        stop = min(count, start + stride)
        offsets = (np.arange(start, stop) * hop)[:, None] + np.arange(window)[None, :]
        spectrum = np.abs(np.fft.rfft(padded[offsets] * taper, axis=1)) ** 2
        levels = 10 * np.log10(spectrum.astype(np.float32) @ onehot + 1e-12)
        peak = float(levels.max())
        floor = max(peak - 70.0, -120.0)
        scaled = np.rint(np.clip((levels.max(axis=0) - floor) / max(1.0, peak - floor), 0, 1) * 255).astype(int)
        columns.append(dict(time_s=round(start * hop / sample_rate + float(offset_s), 5),
                            end_s=round(stop * hop / sample_rate + float(offset_s), 5), levels=scaled.tolist()))
    return dict(offset_s=round(float(offset_s), 5), hop_s=round(hop / sample_rate * stride, 5),
                bands=bands, columns=columns)


def sound_segments(soundtrack, start_s, end_s, max_segments=400):
    """Coarse speech/music/noise/silence hints from existing frame features."""
    import numpy as np
    hop = soundtrack.hop_s
    first = max(0, int((start_s - soundtrack.offset_s) / hop))
    last = min(len(soundtrack.rms_db), int((end_s - soundtrack.offset_s) / hop))
    if last <= first:
        return []
    rms = soundtrack.rms_db[first:last]
    novelty = soundtrack.novelty[first:last]
    silent = soundtrack.silent[first:last]
    high = np.quantile(rms[rms > -70], .75) if np.any(rms > -70) else -50.0
    stride = max(1, math.ceil(len(rms) / max_segments))
    rows = []
    for start in range(0, len(rms), stride):
        chunk = slice(start, min(len(rms), start + stride))
        quiet = float(silent[chunk].mean())
        level = float(rms[chunk].mean())
        change = float(novelty[chunk].mean())
        if quiet >= .6 or level < -50:
            label = "silence"
        elif change > 8 and level > high - 6:
            label = "noise_or_transition"
        elif change < 4 and level > high - 12:
            label = "music_candidate"
        else:
            label = "speech_candidate"
        rows.append(dict(start_s=round((first + start) * hop + soundtrack.offset_s, 4),
                         end_s=round((first + min(len(rms), start + stride)) * hop + soundtrack.offset_s, 4),
                         label=label, rms_db=round(level, 2), novelty_db=round(change, 2),
                         basis="RMS/静音/频谱变化粗分；不是语音识别或音乐分轨"))
    return rows


def no_dialogue_report(value, session, start_frame, end_frame):
    """Dialogue annotations, separately from verified annotation coverage."""
    from shotlab.study import endpoint_us
    tiers = {t["id"]: t["kind"] for t in value.get("tiers", [])}
    events = [e for e in value.get("events", []) if tiers.get(e["tier_id"]) == "dialogue"
              and e["start_frame"] < end_frame and start_frame < e["end_frame"]]

    def union(rows):
        merged = []
        for a, b in sorted((max(start_frame, e["start_frame"]), min(end_frame, e["end_frame"])) for e in rows):
            if b <= a:
                continue
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(b, merged[-1][1])
            else:
                merged.append([a, b])
        return merged

    merged = union(events)
    checked = union(e for e in events if e.get("review_status") in ("confirmed", "revised"))
    gaps = []
    cursor = start_frame
    for a, b in merged:
        if a > cursor:
            gaps.append([cursor, a])
        cursor = max(cursor, b)
    if cursor < end_frame:
        gaps.append([cursor, end_frame])

    def seconds(frame):
        return round(endpoint_us(session.index, int(frame)) / 1e6, 4)

    total = seconds(end_frame) - seconds(start_frame)
    annotated = sum(seconds(b) - seconds(a) for a, b in merged)
    return dict(start_frame=start_frame, end_frame=end_frame, duration_s=round(total, 4),
                reviewed=checked == [[start_frame, end_frame]],
                reviewed_dialogue_s=round(sum(seconds(b) - seconds(a) for a, b in checked), 4),
                dialogue_share=round(annotated / total, 4) if total > 0 else None,
                no_dialogue_s=round(total - annotated, 4),
                intervals=[dict(start_frame=a, end_frame=b, start_s=seconds(a), end_s=seconds(b),
                                duration_s=round(seconds(b) - seconds(a), 4)) for a, b in gaps],
                note="未标注对白不等于没有对白；只有核对过的对白标注覆盖整个范围时才标为已核对")


def region_mask(size, roi, exclusions):
    """Boolean keep-mask in normalized coordinates; exclusions are subtracted."""
    from shotlab.study import validate_roi
    height, width = size
    yy, xx = __import__("numpy").indices((height, width), dtype=float)
    x, y, w, h = validate_roi(roi)
    mask = (xx >= x * width) & (yy >= y * height) & (xx < (x + w) * width) & (yy < (y + h) * height)
    for item in exclusions:
        a, b, c, d = validate_roi(item)
        mask &= ~((xx >= a * width) & (yy >= b * height) & (xx < (a + c) * width) & (yy < (b + d) * height))
    return mask


def color_metrics(rgb, mask):
    """sRGB mean, perceived luminance and saturation over retained pixels."""
    import numpy as np
    pixels = np.asarray(rgb, dtype=np.float32)[mask]
    if not pixels.size:
        return dict(pixels=0, mean_srgb=None, luminance=None, saturation=None)
    mean = pixels.mean(axis=0)
    linear = np.where(mean <= 10.314, mean / 3294.6, ((mean + 14.025) / 269.025) ** 2.4)
    maximum, minimum = pixels.max(axis=1), pixels.min(axis=1)
    saturation = np.divide(maximum - minimum, np.maximum(maximum, 1e-6)).mean()
    return dict(pixels=int(pixels.shape[0]), mean_srgb=[round(float(v), 2) for v in mean],
                luminance=round(float(linear @ [.2126, .7152, .0722]), 5),
                saturation=round(float(saturation), 5))


def composition_guides(size, subject=None):
    """Thirds/center/safe-area lines and normalized subject centre/area/gaze zone."""
    from shotlab.study import validate_roi
    height, width = size
    guides = [dict(kind="thirds", coordinates=[1 / 3, 2 / 3], orientation="vertical"),
              dict(kind="thirds", coordinates=[1 / 3, 2 / 3], orientation="horizontal"),
              dict(kind="center", coordinates=[.5], orientation="vertical"),
              dict(kind="center", coordinates=[.5], orientation="horizontal"),
              dict(kind="safe_area", roi=[.1, .1, .8, .8])]
    if subject is None:
        return dict(guides=guides, subject=None)
    box = validate_roi(subject)
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    zone_x = "左" if cx < 1 / 3 else ("右" if cx > 2 / 3 else "中")
    zone_y = "上" if cy < 1 / 3 else ("下" if cy > 2 / 3 else "中")
    return dict(guides=guides, subject=dict(roi=box, center=[round(cx, 4), round(cy, 4)],
                area=round(w * h, 5), zone=f"{zone_y}{zone_x}", width_px=round(w * width, 1),
                height_px=round(h * height, 1)),
                note="辅助线是阅读参考；主体位置与面积来自人工框选，不代表自动检测")


def phase_template(name):
    templates = {
        "classic": ["anticipation", "action", "buffer", "settle"],
        "impact": ["anticipation", "action", "hold"],
        "minimal": ["hold", "action", "hold"],
    }
    if name not in templates:
        raise ValueError("未知阶段模板")
    return [dict(index=i + 1, phase=phase, label=PHASES[phase]) for i, phase in enumerate(templates[name])]


def adaptive_frame_count(span_frames, base, maximum=6, duration_s=None):
    """Long shots get a bounded number of extra evidence frames."""
    if span_frames <= 0:
        return 1
    seconds = max(0.0, duration_s if duration_s is not None else span_frames / 24)
    extra = math.floor(math.log2(max(1.0, seconds)))
    return max(1, min(span_frames, int(maximum), max(1, int(base)) + extra))


def manual_alignment(anchors):
    """Align two versions by ordered manual anchors; report insertions/deletions.

    Anchors are [first_shot_number, second_shot_number, kind, note] entries.
    `kind` distinguishes composition_simlarity from material_reuse; the method
    never infers reuse from visual similarity alone.
    """
    if not isinstance(anchors, list):
        raise ValueError("人工锚点须为列表")
    if any(not isinstance(a, (list, tuple)) or len(a) < 3
           or type(a[0]) is not int or type(a[1]) is not int for a in anchors):
        raise ValueError("锚点格式须为 [第一版镜头, 第二版镜头, 类型, 备注]")
    rows = []
    previous_a = previous_b = 0
    for anchor in sorted(anchors, key=lambda x: (x[0], x[1])):
        a, b = anchor[:2]
        if a <= previous_a or b <= previous_b or a <= 0 or b <= 0:
            raise ValueError("锚点集数必须严格递增且大于零")
        kind = str(anchor[2])
        if kind not in ALIGNMENT_KINDS:
            raise ValueError("锚点类型须为构图相似或素材复用")
        rows.append(dict(first_shot=a, second_shot=b, inserted_first=max(0, a - previous_a - 1),
                         inserted_second=max(0, b - previous_b - 1),
                         kind=kind,
                         note=text(anchor[3], 500) if len(anchor) > 3 else ""))
        previous_a, previous_b = a, b
    composition = sum(r["kind"] == "composition_similarity" for r in rows)
    reuse = sum(r["kind"] == "material_reuse" for r in rows)
    return dict(anchors=rows, composition_similarity=composition, material_reuse=reuse,
                inserted_or_deleted=sum(r["inserted_first"] + r["inserted_second"] for r in rows),
                note="构图相似与素材复用是不同判断；速度/裁剪差异写入锚点备注，不由哈希自动推断")


def equal_weight_summary(films, inference_unit="film"):
    """One mean per film, then an equal-weight mean across films."""
    import statistics
    if inference_unit not in ("shot", "scene", "film", "episode"):
        raise ValueError("未知推断单位")
    values = []
    for film in films:
        durations = [float(v) for v in film.get("durations", []) if v is not None and math.isfinite(float(v))]
        if durations:
            values.append(dict(label=film.get("label", ""), n=len(durations),
                               mean_s=round(statistics.fmean(durations), 4)))
    return dict(inference_unit=inference_unit, films=values, per_film_equal_weight=True,
                mean_of_film_means_s=round(statistics.fmean([r["mean_s"] for r in values]), 4) if values else None,
                note="分层数据时先按影片/集内聚合，再以每片等权重汇总；镜头不是独立影片样本")


def stratified_summary(films, inference_unit="film"):
    """Explicit hierarchical means from shots → scene/film/episode → films.

    A film contributes one number to the grand mean regardless of its shot
    count. Missing scene labels are a separate, explicitly named stratum.
    """
    import statistics
    levels = ["shot", "scene", "series", "episode", "film"]
    if inference_unit not in ("shot", "scene", "film", "episode"):
        raise ValueError("未知推断单位")
    result = equal_weight_summary(films, inference_unit)
    groups = {}
    for film_index, film in enumerate(films):
        shots = film.get("shots", [])
        if not shots:
            continue
        for shot in shots:
            if shot.get("excluded"):
                continue
            strata = shot.get("strata") or {}
            for level in levels:
                key = str(strata.get(level, film.get("label", "") if level == "film" else "（未填）"))
                identity = shot.get("scene_id") if level == "scene" else key
                group = groups.setdefault((level, film_index, identity),
                                          dict(level=level, film=film.get("label", ""), label=key, n=0, durations=[]))
                group["n"] += 1
                group["durations"].append(float(shot.get("duration_s", 0)))
    rows = []
    for group in groups.values():
        durations = group.pop("durations", [])
        if durations:
            group["mean_s"] = round(statistics.fmean(durations), 4)
        rows.append(group)
    result.update(levels=levels, strata=rows, explicit=True,
                  note="先按 series/episode/film/scene/shot 明确分层；总体仍以每片等权，避免长片因镜头多而获得更大权重")
    return result


def optional_asr_adapter(name, payload=None, params=None):
    """Explicit, user-provided ASR bridge; no bundled models or hard dependencies.

    The adapter never invokes a speech model. A result is accepted only when the
    user supplies segments together with model/version/source/author metadata;
    it is recorded as an imported observation and cannot replace manual speaker
    links or dialogue annotations.
    """
    import importlib.util
    options = {"inaSpeechSegmenter": "inaSpeechSegmenter", "whisperx": "whisperx", "pyannote": "pyannote"}
    if name not in (*options, "none"):
        raise ValueError("未知可选声音适配器")
    params = params or {}
    raw = payload or params.get("segments") or params.get("asr_segments") or []
    installed = False
    if name != "none":
        try:
            installed = importlib.util.find_spec(options[name]) is not None
        except (ImportError, ValueError, AttributeError):
            installed = False
    if not raw:
        return dict(available=False, adapter=name, adapter_installed=installed, segments=[],
                    status="disabled" if name == "none" else ("no_result" if installed else "not_installed"),
                    note="未载入外部识别结果；使用内置 FFmpeg/NumPy 粗分，且不把它当作语音识别"
                         if name == "none" else
                         f"{name} 未提供片段；可按需自行安装并确认模型许可，内置粗分结果仍可使用")
    if name == "none":
        raise ValueError("载入外部片段时请选择来源适配器")
    if not isinstance(raw, list) or len(raw) > 2000:
        raise ValueError("外部 ASR 片段须为不超过 2000 条的列表")
    required = {key: text(params.get(key), 500).strip() for key in ("model", "model_version", "source", "author")}
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError("外部 ASR 结果须标明 " + " / ".join(missing))
    segments = []
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("外部 ASR 片段格式错误")
        try:
            start, end = float(row["start_s"]), float(row["end_s"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("外部 ASR 片段须含数值 start_s / end_s") from error
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("外部 ASR 时间区间无效")
        # Python's json.loads accepts the NaN/Infinity literals, so a value that came
        # from a JSON file still has to be checked before it reaches the project.
        confidence = None
        if row.get("confidence") is not None:
            try:
                confidence = float(row["confidence"])
            except (TypeError, ValueError) as error:
                raise ValueError("外部 ASR 片段的 confidence 须为 0-1 的数值") from error
            if not math.isfinite(confidence):
                raise ValueError("外部 ASR 片段的 confidence 须为 0-1 的有限数值")
        segments.append(dict(start_s=round(start, 4), end_s=round(end, 4),
                             text=text(row.get("text"), 5000), speaker=text(row.get("speaker"), 200),
                             confidence=confidence,
                             label=text(row.get("label"), 100), basis="用户提供；未在本程序内运行模型"))
    return dict(available=True, adapter=name, adapter_installed=installed, segments=segments,
                status="user_provided", **required,
                note="外部结果是独立观察层；模型权重未随 CadensView 打包，也不覆盖人工说话人/对白判断")


def validate_query_clauses(clauses, depth=0):
    if not isinstance(clauses, list) or len(clauses) > 30 or depth > 8:
        raise ValueError("查询条件须为不超过 30 条的列表")
    for clause in clauses:
        if not isinstance(clause, dict) or clause.get("field") not in QUERY_FIELDS:
            raise ValueError("查询条件字段无效")
        if clause["field"] in ("all", "any", "not"):
            children = clause.get("clauses", [])
            if not isinstance(children, list):
                raise ValueError("组合条件的子条件须为列表")
            validate_query_clauses(children, depth + 1)
    return clauses


def query_matches(state, study, metadata, motion_shots, clauses):
    """Apply one combination query to a project-state-like list of shots.

    Shared by the current project and every project in compare.run, so a saved
    query has the same meaning across a series. Fields follow QUERY_FIELDS.
    """
    clauses = validate_query_clauses(clauses)
    tiers = {t.get("id"): t for t in study.get("tiers", [])}
    motion_by_id = {r.get("shot_id"): r for r in motion_shots or []}
    metadata = metadata or {}

    def number(clause):
        try:
            return float(clause.get("value", 0))
        except (TypeError, ValueError) as error:
            raise ValueError("时长/静止率条件须为数值") from error

    def match(clause, row):
        field = clause.get("field")
        if field not in QUERY_FIELDS:
            raise ValueError("查询条件字段无效")
        value = str(clause.get("value", "")).lower()
        if field == "text":
            return value in row["text"]
        if field == "tag":
            return value in {str(t).lower() for t in row["tags"]}
        if field == "scale":
            return value in {str(row["scale"]).lower(), str(row["scale_manual"]).lower()}
        if field == "camera":
            return value in str(row["camera"]).lower()
        if field == "kind":
            wanted = clause.get("kinds") or [value]
            return any(str(v).lower() in row["kinds"] for v in wanted)
        if field in ("duration_min", "stillness_min"):
            actual = row["duration"] if field == "duration_min" else row["stillness"]
            return actual is not None and actual >= number(clause)
        if field in ("duration_max", "stillness_max"):
            actual = row["duration"] if field == "duration_max" else row["stillness"]
            return actual is not None and actual <= number(clause)
        if field == "metadata":
            return value in " ".join(f"{k} {v}" for k, v in metadata.items()).lower()
        children = clause.get("clauses", [])
        results = [match(child, row) for child in children]
        return all(results) if field == "all" else (any(results) if field == "any" else not any(results))

    rows, reasons = [], []
    for shot in state.get("shots", []):
        if shot.get("pending"):
            continue
        events = [e for e in study.get("events", []) if e.get("start_frame", 0) < shot["end"]
                  and shot["start"] < e.get("end_frame", 0)]
        text_blob = " ".join([f"{e.get('label', '')} {e.get('note', '')} {e.get('speaker', '')}" for e in events]
                             + [str(shot.get("note", "")), str(shot.get("cut_number", ""))]).lower()
        timing = motion_by_id.get(shot["id"], {})
        row = dict(text=text_blob[:1000], tags=shot.get("tags", []),
                   scale=(shot.get("scale") or {}).get("label", ""),
                   scale_manual=shot.get("scale_manual") or "",
                   camera=(shot.get("vlm") or {}).get("camera_movement", ""),
                   duration=shot.get("duration_s"), stillness=timing.get("still_share"),
                   kinds={tiers.get(e.get("tier_id"), {}).get("kind", "custom") for e in events})
        results = [match(dict(clause, **row), row) for clause in clauses]
        if all(results):
            rows.append(dict(shot, text=text_blob[:300], stillness=row["stillness"],
                             kinds=sorted(row["kinds"]), events=[e.get("label", "") for e in events]))
        else:
            reasons.append(dict(shot_id=shot["id"], matched=sum(results), total=len(results)))
    return dict(clauses=clauses, shots=rows, matched=len(rows), rejected=len(reasons), reasons=reasons[:200])
