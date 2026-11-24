"""AI 拉片协议：分析包导出、叙事层导入、场景深拆、API 直连与拉片长图。

叙事层（narrative）是独立的 project layer，存"场景 / 剧情线 / 情绪曲线 / 全片结构"，
与逐镜头量测层（切点/景别/运动/声音）互补。场景锚定用镜头序号区间：切点人工修订后
叙事结构不失效。分析包 = prompt.md + schema.json + manifest.json（量测证据）+ sheets/
（拼图）或 frames/（每镜头一图）+ subtitles.srt，交给任意多模态大模型（网页上传或 API
直连），返回 JSON 经容错导入写回。协议纪律参考 lapian-notes（MIT）：身份回带、事实
纪律、枚举不译。

所有写回都走 Session._annotate，与镜头编辑共用同一条撤销链。
"""

import json
import math
import sqlite3
from bisect import bisect_right
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

NARRATIVE_VERSION = 1
PROMPT_VERSION = 1

SCENE_TYPES = ("开场", "起", "承", "转", "合", "冲突", "推进", "转折", "升级", "低谷",
               "高潮", "结尾", "支线", "过渡", "背景", "说明", "结论")
RHYTHM_ROLES = ("setup", "rise", "drop", "pressure", "release", "cooldown", "suspense",
                "peak", "low", "aftertaste")
EMOTION_TYPES = ("curiosity", "humor", "warmth", "romance", "tension", "anxiety", "sadness",
                 "conflict", "hope", "release", "inspiration", "aftertaste")
DEFAULT_STORY_LINES = (
    ("protagonist_action", "主角行动线", "目标 / 选择 / 行动 / 代价"),
    ("antagonist_pressure", "对抗压力线", "阻力与压力如何升级"),
    ("relationship_emotion", "关系情感线", "人物关系的建立与变化"),
    ("world_context", "外部世界线", "世界、背景与规则交代"),
    ("subplot_info", "支线信息线", "支线事件与信息释放"),
)
# 拼图模式参数沿用 lapian-notes domesticAiPackage 的取舍：牺牲密度换可上传。
SHEET_COLUMNS, SHEET_ROWS = 4, 6
SHEET_TILE_W, SHEET_TILE_H = 320, 180
DEFAULT_MAX_IMAGES = SHEET_COLUMNS * SHEET_ROWS * 15  # 360 格封顶
SHEET_JPEG_QUALITY = 82
FRAME_JPEG_QUALITY = 85
SCENE_PALETTE = ((64, 130, 255), (255, 120, 90), (120, 200, 120), (230, 180, 60),
                 (180, 130, 230), (90, 200, 210), (200, 110, 170), (150, 160, 175))

CHAT_OPENER = ("分析这个目录（或压缩包）：读 prompt.md 的任务说明，以 manifest.json 的镜头表和画面材料为证据，"
               "参考 subtitles.srt（如有）。只输出符合 schema.json 结构的一个 JSON，不要输出 JSON 之外的内容。")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require(session):
    if session.project is None or session.index is None:
        raise ValueError("先打开视频并建立索引")


# -- 叙事层读写 ----------------------------------------------------------------

def empty():
    return dict(version=NARRATIVE_VERSION, identity={}, model="", prompt_version=0,
                updated_at="", story_lines=[], scenes=[], curve_points=[], macro={})


def data(session):
    """完整叙事层视图；旧项目没有该 layer 时给空结构而不改动会话。"""
    value = {**empty(), **deepcopy(session.narrative or {})}
    value["version"] = NARRATIVE_VERSION
    if not value["story_lines"]:
        value["story_lines"] = [dict(id=i, title=t, description=d) for i, t, d in DEFAULT_STORY_LINES]
    return value


def narrative_get(session, params, emit=None):
    _require(session)
    return data(session)


# -- 时间与文本辅助 ------------------------------------------------------------

def shot_seconds(session, frame):
    """源时间轴秒数：与播放器 -ss、SRT 字幕同一坐标系（相对首个索引帧）。"""
    index = session.index
    return (index.us(frame) - index.us(0)) / 1e6


def timecode(seconds):
    seconds = max(0.0, float(seconds))
    minutes, rest = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}:{minutes:02d}:{rest:04.1f}"
    return f"{minutes:02d}:{rest:04.1f}"


def file_timecode(seconds):
    """文件名安全的时间码：01m23s4（小时位拼作 h）。"""
    seconds = max(0.0, float(seconds))
    minutes, rest = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    tenth = min(int((rest % 1) * 10), 9)
    if hours:
        return f"{hours}h{minutes:02d}m{int(rest):02d}s{tenth}"
    return f"{minutes:02d}m{int(rest):02d}s{tenth}"


def parse_time(value):
    """模型返回的时间可能是秒数（数字）或 '52:30' / '1:02:03.5' 时间码；非有限值一律拒收。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    if not isinstance(value, str):
        return None
    parts = value.strip().replace(",", ".").split(":")
    if not parts or len(parts) > 3 or not parts[0]:
        return None
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    if any(not math.isfinite(n) or n < 0 for n in numbers) \
            or (len(numbers) > 1 and any(n >= 60 for n in numbers[1:])):
        return None
    while len(numbers) < 3:
        numbers.insert(0, 0.0)
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def build_identity(session):
    source = Path(session.video_path).name if session.video_path else ""
    title = session.metadata.get("title") or Path(source).stem
    completed = session.project.completed_count
    last = max(session.project.frame_count - 1, 0)
    duration = (session.index.end_us(last) - session.index.us(0)) / 1e6 if session.index else 0.0
    return {"title": title, "source": source, "shot_count": completed,
            "duration_s": round(duration, 3)}


# -- 证据清单（manifest） -------------------------------------------------------

def _vlm_evidence(row):
    vlm = row.get("vlm") or {}
    keys = ("shot_scale", "camera_movement", "camera_angle", "subjects", "setting",
            "content_type", "special_segment", "effects", "description", "confidence")
    return {k: vlm[k] for k in keys if vlm.get(k) is not None}


def _subtitle_text(cues, start_s, end_s):
    return " / ".join(cue["text"].replace("\n", " ")
                      for cue in cues if cue["start_us"] / 1e6 < end_s and cue["end_us"] / 1e6 > start_s)


def _measured(session, params):
    """有缓存且范围匹配的量测才进证据；过期或缺失不硬凑。"""
    measured = {}
    try:
        scope = session.analysis_scope(params)
        key = session._motion_key(scope)
        if session.motion is not None and session.motion_key == key:
            measured["motion"] = {row["shot_number"]: row for row in session.motion["shots"]}
        if session.audio is not None and session.audio_key == key:
            measured["audio"] = {row["shot_number"]: row for row in session.audio["shots"]}
    except ValueError:
        pass
    return measured


def _shot_seconds_row(session, row):
    index = session.index
    start_s = (index.us(row["start"]) - index.us(0)) / 1e6
    last = max(min(row["end"] - 1, session.project.frame_count - 1), row["start"])
    end_s = (index.end_us(last) - index.us(0)) / 1e6
    return start_s, max(end_s, start_s)


def build_manifest(session, params):
    rows = session.scoped_shots(params)
    measured = _measured(session, params)
    cues = session.subtitle_cues or []
    shots = []
    for row in rows:
        start_s, end_s = _shot_seconds_row(session, row)
        motion = measured.get("motion", {}).get(row["number"] + 1, {})
        audio = measured.get("audio", {}).get(row["number"] + 1, {})
        entry = {
            "number": row["number"] + 1,
            "shot_id": row["id"],
            "start_frame": row["start"], "end_frame": row["end"],
            "start_s": round(start_s, 3), "end_s": round(end_s, 3),
            "duration_s": row["duration_s"],
            "scale": row.get("scale_manual") or (row.get("scale") or {}).get("label"),
            "scale_source": "manual" if row.get("scale_manual") else
                            ("model" if (row.get("scale") or {}).get("label") else None),
            "scale_confidence": (row.get("scale") or {}).get("confidence"),
            "tags": row.get("tags", []),
            "cut_number": row.get("cut_number", ""),
            "region": row.get("region"),
            "dialogue": _subtitle_text(cues, start_s, end_s) or None,
        }
        entry.update(_vlm_evidence(row))
        for key, source in (("motion", motion), ("koma", motion),
                            ("loudness_lufs", audio), ("silence_share", audio)):
            if source.get(key) is not None:
                entry[key] = source[key]
        shots.append(entry)
    return {"identity": build_identity(session), "shots": shots,
            "evidence": {"subtitles": bool(cues),
                         "vlm_annotated": sum(1 for r in rows if r.get("vlm")),
                         "motion": "current" if "motion" in measured else "absent",
                         "audio": "current" if "audio" in measured else "absent"}}


# -- 帧采集（拼图 / 原图） ------------------------------------------------------

def _sample_rows(rows, limit):
    if len(rows) <= limit:
        return list(rows), None
    # limit>=2 时等距公式天然含首末镜头；limit==1 只取首镜头。
    picked = [0] if limit == 1 else sorted(
        {round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)})
    return [rows[p] for p in picked], \
        f"镜头 {len(rows)} 个，超出图像上限 {limit}，均匀采样 {len(picked)} 个"


def _shot_label(row, session):
    return f"S{row['number'] + 1:04d} {timecode(shot_seconds(session, row['start']))}"


def _decode_thumbs(session, rows, width, cancel=None, progress=None):
    """逐镜头中帧缩略图（最近邻缩放），与 rpc._scoped_thumbs 同一套路。"""
    from concurrent.futures import CancelledError

    import numpy as np
    from t0.media import FrameReader
    reader = FrameReader(session.index)
    thumbs = []
    try:
        for done, row in enumerate(rows, 1):
            if cancel is not None and cancel.is_set():
                raise CancelledError("拉片包导出已取消")
            mid = min((row["start"] + row["end"]) // 2, len(session.index.pts) - 1)
            array = reader.read(mid).to_ndarray(format="rgb24")
            h, w = array.shape[:2]
            th = int(h * width / w)
            if th > 0:
                ys = (np.arange(th) * h // th).clip(0, h - 1)
                xs = (np.arange(width) * w // width).clip(0, w - 1)
                thumbs.append((array[ys][:, xs], row))
            if progress is not None:
                progress(done, len(rows))
    finally:
        reader.close()
    if not thumbs:
        raise ValueError("没有可导出的镜头画面")
    return thumbs


def _tile_image(array, width, height, label):
    """等比缩放后居中贴到固定底板，左上角烧录镜头号与时间码。"""
    from PIL import Image, ImageDraw

    from shotlab.fonts import ui_font
    image = Image.fromarray(array)
    scale = min(width / image.width, height / image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(new_size)
    board = Image.new("RGB", (width, height), (12, 12, 16))
    board.paste(resized, ((width - new_size[0]) // 2, (height - new_size[1]) // 2))
    ImageDraw.Draw(board).text((6, 4), label, fill=(255, 230, 80), font=ui_font(13),
                               stroke_width=2, stroke_fill=(0, 0, 0))
    return board


def _write_sheets(out_dir, thumbs, labels):
    """4×6 拼图组；返回 (文件名列表, {0 基镜头号: (文件名, 格序)})。"""
    from PIL import Image
    per_sheet = SHEET_COLUMNS * SHEET_ROWS
    (out_dir / "sheets").mkdir(parents=True, exist_ok=True)
    names, mapping = [], {}
    for sheet_index in range(0, len(thumbs), per_sheet):
        part = thumbs[sheet_index:sheet_index + per_sheet]
        sheet_no = len(names) + 1
        name = f"sheet-{sheet_no:02d}.jpg"
        w = SHEET_COLUMNS * (SHEET_TILE_W + 4) + 4
        h = SHEET_ROWS * (SHEET_TILE_H + 4) + 4
        sheet = Image.new("RGB", (w, h), (16, 16, 20))
        for tile, (array, row) in enumerate(part):
            x = tile % SHEET_COLUMNS * (SHEET_TILE_W + 4) + 4
            y = tile // SHEET_COLUMNS * (SHEET_TILE_H + 4) + 4
            sheet.paste(_tile_image(array, SHEET_TILE_W, SHEET_TILE_H, labels[row["number"]]), (x, y))
            mapping[row["number"]] = (f"sheets/{name}", tile + 1)
        sheet.save(str(out_dir / "sheets" / name), "JPEG", quality=SHEET_JPEG_QUALITY)
        names.append(f"sheets/{name}")
    return names, mapping


def _write_frames(out_dir, thumbs, labels, codes):
    """每镜头一张代表帧；返回 {0 基镜头号: 文件名}。"""
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    mapping = {}
    for array, row in thumbs:
        name = f"shot-{row['number'] + 1:04d}_{codes[row['number']]}.jpg"
        image = _tile_image(array, SHEET_TILE_W, SHEET_TILE_H, labels[row["number"]])
        image.save(str(out_dir / "frames" / name), "JPEG", quality=FRAME_JPEG_QUALITY)
        mapping[row["number"]] = f"frames/{name}"
    return mapping


# -- 分析包导出 ----------------------------------------------------------------

def _clear_old_images(out_dir, folder, pattern):
    """Re-exporting into the same folder must not leave images of an earlier, larger package
    behind: the model is told to analyse the whole directory and would see stale frames."""
    target = Path(out_dir) / folder
    if target.is_dir():
        for stale in target.glob(pattern):
            stale.unlink(missing_ok=True)


def export_package(session, params, cancel=None, progress=None):
    """导出 AI 拉片分析包到目录；mode=sheets（拼图，省流量）或 frames（每镜头一图）。"""
    _require(session)
    out_dir = Path(params["out_dir"])
    mode = params.get("mode", "sheets")
    if mode not in ("sheets", "frames"):
        raise ValueError("mode 须为 sheets 或 frames")
    max_images = max(1, int(params.get("max_images", DEFAULT_MAX_IMAGES)))

    rows = session.scoped_shots(params)
    manifest = build_manifest(session, params)
    sampled, sampling_note = _sample_rows(rows, max_images)
    labels = {row["number"]: _shot_label(row, session) for row in rows}
    codes = {row["number"]: file_timecode(shot_seconds(session, row["start"])) for row in rows}
    thumbs = _decode_thumbs(session, sampled, SHEET_TILE_W, cancel, progress)

    if mode == "sheets":
        _clear_old_images(out_dir, "sheets", "sheet-*.jpg")
        names, mapping = _write_sheets(out_dir, thumbs, labels)
        for entry in manifest["shots"]:
            hit = mapping.get(entry["number"] - 1)
            if hit:
                entry["image"], entry["tile"] = hit
        image_files = len(names)
    else:
        _clear_old_images(out_dir, "frames", "shot-*.jpg")
        mapping = _write_frames(out_dir, thumbs, labels, codes)
        for entry in manifest["shots"]:
            hit = mapping.get(entry["number"] - 1)
            if hit:
                entry["image"] = hit
        image_files = len(mapping)
    manifest["evidence"]["image_mode"] = mode
    manifest["evidence"]["image_count"] = len(thumbs)
    if sampling_note:
        manifest["evidence"]["sampling"] = sampling_note

    has_subtitles = bool(session.subtitle_cues)
    (out_dir / "prompt.md").write_text(build_prompt(manifest), encoding="utf-8")
    (out_dir / "schema.json").write_text(
        json.dumps(build_return_schema(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if has_subtitles:
        (out_dir / "subtitles.srt").write_text(build_srt(session), encoding="utf-8")

    payload = {"out_dir": str(out_dir.resolve()), "mode": mode, "shots": len(rows),
               "images": len(thumbs), "image_files": image_files,
               "subtitles": has_subtitles, "sampling": sampling_note,
               "chat_opener": CHAT_OPENER, "prompt_version": PROMPT_VERSION}
    return payload


def build_srt(session):
    lines = []
    for index, cue in enumerate(session.subtitle_cues, 1):
        # A blank line inside a cue ends the cue for every SRT reader; dialogue
        # events typed in the study layer may contain them.
        text = "\n".join(line for line in str(cue["text"]).splitlines() if line.strip())
        lines += [str(index), f"{_srt_time(cue['start_us'])} --> {_srt_time(cue['end_us'])}",
                  text, ""]
    return "\n".join(lines)


def _srt_time(us):
    ms = max(0, us // 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


# -- Prompt 与返回结构 ----------------------------------------------------------

def build_prompt(manifest, deep=None):
    identity, evidence = manifest["identity"], manifest["evidence"]
    if deep is not None:
        lines = _deep_prompt_lines(deep)
    else:
        image_clause = {
            "sheets": "画面拼图若干张：每格是一个镜头的代表帧，左上角烧录镜头号（S 编号）与时间码。",
            "frames": "frames/ 内每镜头一张代表帧，文件名与图内左上角都有镜头号与时间码。",
        }.get(evidence.get("image_mode"), "没有画面图像。")
        lines = [
            "你是电影/动画拉片助手。手头材料：prompt.md（本文件）、manifest.json（逐镜头量测证据表）、"
            + ("subtitles.srt（全片字幕）。" if evidence.get("subtitles") else "没有字幕。") + image_clause,
            "",
            "任务：把整部作品按真实剧情变化划分为若干场景（scene），并为每个场景、每条剧情线、"
            "全片结构与观众情绪曲线产出结构化拉片笔记。",
        ]
    numbers = [shot["number"] for shot in manifest.get("shots", []) if isinstance(shot.get("number"), int)]
    if numbers and (numbers[0] != 1 or numbers[-1] != identity["shot_count"] or len(numbers) != identity["shot_count"]):
        # A scoped package (marked or excluded shots) lists a subset: the model must not be
        # told the range is 1..shot_count, and scene anchors have to stay on listed shots.
        scope_line = (f"本次只分析 manifest.shots 列出的 {len(numbers)} 个镜头（全片共 {identity['shot_count']} 个；"
                      "场景的 start_shot/end_shot 只能使用其中出现过的镜头号）")
    else:
        scope_line = f"镜头数：{identity['shot_count']}（manifest.shots 的 number 范围 1-{identity['shot_count']}）"
    lines += [
        "",
        "核心纪律：",
        "1. 场景标题必须是具体剧情事件（如「小鱼在天台摊牌」），禁止「开场建立」「冲突升级」等模板标题。",
        "2. 事实纪律：人物名、地名、身份、设定必须以字幕与画面证据为准；manifest 里的量测字段"
        "（scale/duration_s/motion/loudness_lufs 等）是仪器数据，不得改写；你的叙事判断与量测矛盾时，"
        "在 techniques 里说明你看到的画面依据。",
        "3. 场景锚定：scenes 的 start_shot/end_shot 使用 manifest.shots 的 number（1 起，含端点）；"
        "相邻场景首尾相接，一起覆盖全部镜头，不要重叠、不要留空洞。",
        "4. key_beats 每条前标注时间码（如「12:03 小鱼说出真相」），便于人工回看核对。",
        "5. 枚举值（type / rhythm_role / emotion_type / primary_line 等）必须原样使用给定值，不要翻译、"
        "不要自造；描述性文本一律用简体中文。",
        "6. filmIdentity 三项（title/source/shot_count）必须原样带回，用于导入校验。",
    ]
    if evidence.get("subtitles") is False:
        lines.append("7. 没有字幕：对白无法获得，无法确认的对白内容不要编造。")
    if not evidence.get("image_mode"):
        lines.append("8. 没有画面：完全依据 manifest 量测判断，techniques 没把握就留空。")
    if evidence.get("sampling"):
        lines.append(f"注意：{evidence['sampling']}；未采样镜头只能依据 manifest 与字幕判断。")
    lines += [
        "",
        "剧情线（story_lines）规则：",
        "- 默认五条线已预置（见 schema）；建议为这部作品定制各线标题（title 可改，id 与数量不改）。",
        "- 每个场景的 primary_line 必须是 story_lines 里定义的 id；跨线场景把 shared_lines 写全（含 primary_line）。",
        "",
        "情绪曲线（curve_points）规则：",
        "- intensity 是观众激活度（0-100），不是音量也不是重要度：安静但扣人心弦的场面（告白、对峙、"
        "无声崩溃）是高强度低效价；禁止强度随时间机械上升。",
        "- valence 是情绪方向（-100 到 100）。全片 12-20 个点，不是每个场景一个点。",
        "- 必须包含局部起落：至少 3 次明显回落/压抑、2 次抬升、1 个低谷、1-3 个峰值、高潮后的余韵。",
        "- rhythm_role ∈ " + "/".join(RHYTHM_ROLES) + "；emotion_type ∈ " + "/".join(EMOTION_TYPES) + "。",
        "- why 说明观众体验为什么在这里变化、服务于哪个节奏角色。",
        "",
        f"影片名：{identity['title']}",
        f"源视频文件名：{identity['source']}",
        scope_line,
        f"片长：{identity['duration_s']} 秒",
        "",
        "只输出一个 JSON（顶层含 filmIdentity / story_lines / scenes / curve_points / macro），"
        "不要输出 JSON 之外的说明。结构示例即 schema.json。",
    ]
    return "\n".join(lines)


def _deep_prompt_lines(deep):
    scene = deep["scene"]
    return [
        f"你是电影/动画拉片助手。这次只深拆一个场景：S{scene['start_shot']:04d}-S{scene['end_shot']:04d}"
        f"「{scene['title']}」（现有粗稿见 scene.json，你的任务是替换成逐镜头精细版）。",
        "材料：本文件、scene.json（场景粗稿）、manifest.json（本场景各镜头的量测证据与台词）、"
        "frames/ 内本场景每个镜头的代表帧。",
        "",
        "深拆要求：",
        "1. shot_cards 是核心产出：按镜头顺序给每个镜头一张卡（action 画面里实际发生什么 / dialogue "
        "引用台词原文，无则空串 / technique 这一格的视听手法，没把握留空）。",
        "2. 场景级字段（title/function/techniques/key_beats）按本场景精写；techniques 至少 3 条，"
        "逐镜头观察构图、景别、机位与剪辑，每条标注镜头号。",
        "3. 事实纪律与量测不服从规则同全片分析；shot_cards 的 shot 必须是 manifest 里的镜头号。",
    ]


def build_return_schema(manifest):
    identity = manifest["identity"]
    return {
        "filmIdentity": {"title": identity["title"], "source": identity["source"],
                         "shot_count": identity["shot_count"]},
        "story_lines": [{"id": line[0], "title": f"为这部作品定制的线名（默认：{line[1]}）",
                         "description": line[2]} for line in DEFAULT_STORY_LINES],
        "scenes": [{
            "start_shot": 1, "end_shot": 12,
            "title": "具体剧情事件标题",
            "type": "/".join(SCENE_TYPES) + " 之一",
            "function": "这一场在全片结构、人物关系、信息释放与节奏中的作用",
            "primary_line": "protagonist_action",
            "shared_lines": ["protagonist_action"],
            "techniques": "逐镜头视听手法，换行分隔，每条标注镜头号或时间码",
            "key_beats": "12:03 节拍一\n14:40 节拍二（每条带时间码）",
            "confidence": 0.8,
        }],
        "curve_points": [{
            "time": 0, "intensity": 35, "valence": 0,
            "emotion_type": "curiosity", "rhythm_role": "setup",
            "title": "观众体验变化点", "why": "为什么在这里变化",
        }],
        "macro": {
            "overall_structure": "全片结构判断",
            "narrative_strategy": "主角目标、阻力、利害",
            "rhythm_pattern": "关键转折、高潮、解决",
            "information_strategy": "信息释放、悬念、反转",
            "core_intent": "人物变化与主题选择",
            "writing_lessons": ["可复用写法"],
        },
    }


# -- 容错解析 -------------------------------------------------------------------

def extract_json(text):
    """从模型回复里括号配平地抠出第一个可解析 JSON 对象（容忍围栏与前后杂文）。"""
    start = text.find("{")
    # Every failed start rescans to the end, so a malformed reply is O(n²) otherwise and
    # would freeze the sidecar's request loop for minutes.
    budget = 20_000_000
    while start != -1:
        depth, in_string, escape = 0, False, False
        for position in range(start, len(text)):
            budget -= 1
            if budget < 0:
                raise ValueError("回复里找不到完整的 JSON 对象（内容过长或已损坏），请重新获取模型输出")
            char = text[position]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:position + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return json.loads(text)  # 让原始异常冒泡，信息更完整


def unwrap_payload(value, depth=0):
    """模型可能把结果再包一层（result/data/analysis/单键容器），剥到分析对象为止。"""
    markers = ("scenes", "segments", "curve_points", "story_lines", "macro", "shot_cards")
    if not isinstance(value, dict) or depth > 3:
        return value
    if any(key in value for key in markers):
        return value
    children = [v for v in value.values() if isinstance(v, dict)]
    if len(children) == 1:
        unwrapped = unwrap_payload(children[0], depth + 1)
        if isinstance(unwrapped, dict) and any(k in unwrapped for k in markers):
            return unwrapped
    return value


# -- 归一化与导入 ----------------------------------------------------------------

def _first(payload, *keys):
    if not isinstance(payload, dict):
        return None
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _shot_at_time(starts, seconds):
    return min(max(bisect_right(starts, seconds + 1e-6) - 1, 0), len(starts) - 1)


def _scene_bounds(item, starts):
    """双通道定位：镜头号优先，缺省按时间码映射到覆盖镜头。返回 (start, end, method, warnings)。"""
    warnings = []
    start = _first(item, "start_shot", "shot_start", "startShot", "first_shot")
    end = _first(item, "end_shot", "shot_end", "endShot", "last_shot")
    if isinstance(start, str) and start.strip().isdigit():
        start = int(start)
    if isinstance(end, str) and end.strip().isdigit():
        end = int(end)
    method = "shots"
    if not isinstance(start, int) or not isinstance(end, int):
        method = "time"
        start_t = parse_time(_first(item, "start_time", "startTime", "start"))
        end_t = parse_time(_first(item, "end_time", "endTime", "end"))
        if start_t is None or end_t is None:
            return None, None, method, ["场景缺少可用的镜头号或时间定位，已丢弃"]
        start = _shot_at_time(starts, start_t) + 1
        end = _shot_at_time(starts, max(start_t, end_t - 1e-6)) + 1
        warnings.append(f"场景「{str(item.get('title', ''))[:20]}」按时间码定位为镜头 {start}-{end}")
    limit = len(starts)
    clamped = False
    if start < 1:
        start, clamped = 1, True
    if end > limit:
        end, clamped = limit, True
    if end < start:
        start = end = min(start, limit)
        clamped = True
    if clamped:
        warnings.append(f"场景「{str(item.get('title', ''))[:20]}」镜头范围越界，已收敛为 {start}-{end}")
    return start, end, method, warnings


def _dict_items(value):
    """数组/对象里混入的非字典项（模型偶发输出）跳过并计数，不让整次导入崩掉。

    容器本身不是列表/字典（如 scenes: 2.5）时按全部无效处理；字典容器取其值
    （scenes: {"s1": {...}} 这种按 id 键的形态也认）。
    """
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list):
        return [], 1
    items, skipped = [], 0
    for item in value:
        if isinstance(item, dict):
            items.append(item)
        else:
            skipped += 1
    return items, skipped


def normalize_analysis(session, payload, params=None):
    """把模型 JSON 归一成叙事层结构；只读会话，不写回。返回 (narrative, warnings, preview)。"""
    if not isinstance(payload, dict):
        raise ValueError("返回的 JSON 顶层不是对象，无法导入")
    rows = session.scoped_shots(params or {})
    if not rows:
        raise ValueError("没有已完成镜头，不能导入拉片结果")
    starts = [shot_seconds(session, row["start"]) for row in rows]
    identity = build_identity(session)
    warnings = []

    given = _first(payload, "filmIdentity", "movieIdentity", "identity", "project")
    if not isinstance(given, dict):
        given = {}
    given_source = str(given.get("source") or given.get("sourceVideoName") or "")
    given_title = str(given.get("title") or given.get("filmTitle") or "")
    given_count = given.get("shot_count", given.get("shotCount"))
    if given_source and given_source != identity["source"]:
        raise ValueError(f"影片不匹配：结果属于「{given_source}」，当前项目是「{identity['source']}」")
    if isinstance(given_count, int) and given_count != identity["shot_count"]:
        raise ValueError(f"镜头数不匹配：结果 {given_count}，当前项目 {identity['shot_count']}（切点可能已修订）")
    if given_title and given_title != identity["title"]:
        warnings.append(f"影片名不一致（结果「{given_title}」/ 项目「{identity['title']}」），已按源文件名通过")

    lines = []
    line_items, junk_lines = _dict_items(_first(payload, "story_lines", "storyLines", "lines"))
    if junk_lines:
        warnings.append(f"story_lines 跳过 {junk_lines} 个非对象项")
    for item in line_items:
        line_id = str(item.get("id") or "").strip()
        if line_id:
            lines.append({"id": line_id, "title": str(item.get("title", ""))[:120],
                          "description": str(item.get("description", ""))[:500]})
    if not lines:
        lines = [dict(id=i, title=t, description=d) for i, t, d in DEFAULT_STORY_LINES]
    line_ids = {line["id"] for line in lines}

    scenes, unknown_types, used_ids = [], 0, set()
    scene_items, junk_scenes = _dict_items(_first(payload, "scenes", "segments"))
    if junk_scenes:
        warnings.append(f"scenes 跳过 {junk_scenes} 个非对象项")
    for item in scene_items:
        start, end, method, bound_warnings = _scene_bounds(item, starts)
        warnings += bound_warnings
        if start is None:
            continue
        scene_type = str(item.get("type", "") or "过渡")
        if scene_type not in SCENE_TYPES:
            unknown_types += 1
        primary = str(item.get("primary_line", "") or "").strip()
        if primary and primary not in line_ids:
            warnings.append(f"场景「{str(item.get('title', ''))[:20]}」primary_line「{primary}」不在剧情线中")
            primary = ""
        shared = [str(s) for s in item.get("shared_lines", []) or [] if str(s) in line_ids]
        scenes.append({
            # 确定性 id：同一分析的重复导入幂等（write_notes 的替换标记依赖它）
            "id": _scene_id(item, start, end, used_ids),
            "start_shot": start, "end_shot": end,
            "start_s": round(starts[start - 1], 3),
            "end_s": round(_shot_seconds_row(session, rows[end - 1])[1], 3),
            "anchored_by": method,
            "title": str(item.get("title", "")).strip()[:200] or f"场景 {start}-{end}",
            "type": scene_type,
            "function": str(item.get("function") or item.get("segment_function") or "")[:2000],
            "primary_line": primary or None,
            "shared_lines": shared or None,
            "techniques": str(item.get("techniques", "") or "")[:3000],
            "key_beats": str(item.get("key_beats") or item.get("keyBeats") or "")[:3000],
            "confidence": _confidence(item.get("confidence")),
        })
    scenes.sort(key=lambda s: (s["start_shot"], s["end_shot"]))
    for previous, current in zip(scenes, scenes[1:]):
        if current["start_shot"] <= previous["end_shot"]:
            warnings.append(f"场景「{current['title'][:20]}」与上一场景镜头重叠")
    if scenes and (scenes[0]["start_shot"] != 1 or scenes[-1]["end_shot"] != len(rows)):
        warnings.append("场景未完整覆盖全部镜头（首尾有缺口）")

    curve, unknown_emotions, unknown_roles = [], 0, 0
    curve_items, junk_curve = _dict_items(
        _first(payload, "curve_points", "audienceCurvePoints", "emotionPoints", "curve"))
    if junk_curve:
        warnings.append(f"curve_points 跳过 {junk_curve} 个非对象项")
    for item in curve_items:
        time = parse_time(item.get("time"))
        if time is None:
            warnings.append(f"曲线点「{str(item.get('title', ''))[:20]}」时间无效，已丢弃")
            continue
        emotion = str(item.get("emotion_type") or item.get("emotionType") or "")
        role = str(item.get("rhythm_role") or item.get("rhythmRole") or "")
        unknown_emotions += bool(emotion and emotion not in EMOTION_TYPES)
        unknown_roles += bool(role and role not in RHYTHM_ROLES)
        curve.append({
            "time": round(min(max(time, 0.0), identity["duration_s"]), 3),
            "intensity": _clamp(item.get("intensity"), 0, 100),
            "valence": _clamp(item.get("valence"), -100, 100),
            "emotion_type": emotion or None, "rhythm_role": role or None,
            "title": str(item.get("title", "")).strip()[:200],
            "why": str(item.get("why") or item.get("description") or "")[:1000],
        })
    curve.sort(key=lambda p: p["time"])

    macro_raw = _first(payload, "macro", "macroAnalysis", "macro_analysis")
    if not isinstance(macro_raw, dict):
        macro_raw = {}
    macro = {key: str(macro_raw.get(key, "") or "")[:2000]
             for key in ("overall_structure", "narrative_strategy", "rhythm_pattern",
                         "information_strategy", "core_intent")}
    lessons = macro_raw.get("writing_lessons", []) or []
    macro["writing_lessons"] = [str(x)[:500] for x in ([lessons] if isinstance(lessons, str) else lessons)][:20]

    narrative = dict(empty(), identity=identity, story_lines=lines, scenes=scenes,
                     curve_points=curve, macro=macro)
    preview = {"scenes": len(scenes), "story_lines": len(lines), "curve_points": len(curve),
               "unknown_scene_types": unknown_types, "unknown_emotion_types": unknown_emotions,
               "unknown_rhythm_roles": unknown_roles,
               "coverage": [f"{s['start_shot']}-{s['end_shot']}" for s in scenes[:8]],
               "warnings": warnings}
    return narrative, warnings, preview


def _scene_identity(start, end, title):
    import hashlib
    return hashlib.sha1(f"{start}-{end}-{title}".encode("utf-8")).hexdigest()[:12]


def _scene_id(item, start, end, used):
    """Deterministic scene id. A model-supplied id is kept only when it is unique and
    single-line: the id becomes a note marker in write_notes and deep_import replaces
    scenes by id, so a duplicated id would merge or overwrite another scene."""
    given = "".join(ch for ch in str(item.get("id") or "") if not ch.isspace())[:64]
    base = given or _scene_identity(start, end, item.get("title", ""))
    identity, serial = base, 2
    while identity in used:
        identity, serial = f"{base}-{serial}", serial + 1
    used.add(identity)
    return identity


def _confidence(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(number, 0.0), 1.0) if math.isfinite(number) else None


def _clamp(value, low, high):
    """限幅并拒绝非有限值：Python 的 json 会收下 NaN/Infinity 字面量，
    一旦写进叙事层，持久化 JSON 里的 NaN 会让 Rust serde_json 拒载整个项目。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(number, low), high) if math.isfinite(number) else None


def _commit_import(session, final, write_notes):
    """一条撤销记录同时覆盖叙事层与（可选）镜头笔记；任一步失败整体回滚。

    不能走 Session._annotate：它在 mutate 之后才快照 project（前后同值），笔记改动会
    撤销不回去；这里按 _edit 的口径先存 before，再合并 annotations extra。
    """
    import copy
    before_project = copy.deepcopy(session.project)
    before_annotations = session.annotation_state()
    try:
        session.narrative = final
        if write_notes:
            _write_scene_notes(session, final["scenes"])
        session.project.validate()
    except BaseException:
        session.project = before_project
        session.restore_annotations(before_annotations)
        raise
    session.history.record(before_project, copy.deepcopy(session.project), "导入 AI 拉片", None,
                           extra=(before_annotations, session.annotation_state()))
    session.invalidate_results()  # 笔记属于研究导出数据，统计缓存一并失效
    session.dirty = True


def import_analysis(session, params, emit=None):
    """导入模型返回的 JSON（params: text|path, dry_run, write_notes, model）。"""
    _require(session)
    text = params.get("text")
    if not text and params.get("path"):
        text = Path(params["path"]).read_text(encoding="utf-8")
    if not text or not str(text).strip():
        raise ValueError("缺少要导入的 JSON（text 或 path）")
    payload = unwrap_payload(extract_json(str(text)))
    narrative, _warnings, preview = normalize_analysis(session, payload, params)
    if params.get("dry_run"):
        return {"dry_run": True, "preview": preview}
    model = str(params.get("model", "") or "")
    final = dict(narrative, model=model, prompt_version=PROMPT_VERSION, updated_at=now())
    _commit_import(session, final, bool(params.get("write_notes")))
    return {"imported": True, "preview": preview, "model": model}


def _write_scene_notes(session, scenes):
    """把场景标题写进各镜头笔记；同场景 id 的旧行被替换，避免重复导入堆叠。"""
    shots = {shot.id: shot for shot in session.project.shots}
    numbers = {row["number"] + 1: row["id"] for row in session.state()["shots"]}
    for scene in scenes:
        marker = f"〔AI#{scene['id']}〕"
        for number in range(scene["start_shot"], scene["end_shot"] + 1):
            shot = shots.get(numbers.get(number))
            if shot is None:
                continue
            kept = [line for line in shot.note.split("\n") if not line.startswith(marker)]
            session.project.set_note(numbers[number], "\n".join(kept + [marker + scene["title"]]))


# -- 场景深拆（P2） --------------------------------------------------------------

def _pick_scene(narrative, scene_id=None, scene_index=None):
    scenes = narrative["scenes"]
    if scene_id:
        for scene in scenes:
            if scene["id"] == str(scene_id):
                return scene
        raise ValueError(f"场景 {scene_id} 不存在")
    if scene_index is not None:
        if not 0 <= int(scene_index) < len(scenes):
            raise ValueError(f"场景序号须在 0-{len(scenes) - 1}")
        return scenes[int(scene_index)]
    raise ValueError("给出 scene_id 或 scene_index")


def export_deep(session, params, cancel=None, progress=None):
    """单个场景的深拆包：该场景每镜头一图 + 台词 + 场景粗稿。"""
    _require(session)
    narrative = data(session)
    if not narrative.get("identity"):
        raise ValueError("叙事层为空，请先导入全片分析")
    scene = _pick_scene(narrative, params.get("scene_id"), params.get("scene_index"))
    rows = [row for row in session.scoped_shots({})
            if scene["start_shot"] <= row["number"] + 1 <= scene["end_shot"]]
    if not rows:
        raise ValueError("场景范围内没有已完成镜头（切点可能已大改，请重跑全片分析）")
    out_dir = Path(params["out_dir"])
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    _clear_old_images(out_dir, "frames", "shot-*.jpg")

    sub_manifest = build_manifest(session, {"shot_ids": [row["id"] for row in rows]})
    labels = {row["number"]: _shot_label(row, session) for row in rows}
    for array, row in _decode_thumbs(session, rows, 480, cancel, progress):
        _tile_image(array, 480, 270, labels[row["number"]]).save(
            str(out_dir / "frames" / f"shot-{row['number'] + 1:04d}.jpg"),
            "JPEG", quality=FRAME_JPEG_QUALITY)

    scene_json = {"scene": {k: scene[k] for k in ("id", "start_shot", "end_shot", "title", "type",
                                                  "function", "techniques", "key_beats")}}
    (out_dir / "prompt.md").write_text(build_prompt(sub_manifest, deep={"scene": scene}),
                                       encoding="utf-8")
    (out_dir / "scene.json").write_text(json.dumps(scene_json, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(sub_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "schema.json").write_text(
        json.dumps(build_deep_schema(scene), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"out_dir": str(out_dir.resolve()), "scene": scene_json["scene"],
            "shots": len(rows), "prompt_version": PROMPT_VERSION}


def build_deep_schema(scene):
    return {
        "filmIdentity": "原样带回 manifest.identity",
        "scene_id": scene["id"],
        "title": "精写后的场景标题", "type": "枚举同全片分析", "function": "本场景作用",
        "techniques": "逐镜头视听手法（每条标注镜头号）", "key_beats": "每条带时间码",
        "shot_cards": [{"shot": 12, "action": "画面里实际发生什么",
                        "dialogue": "引用台词原文，无则空串", "technique": "本格视听手法"}],
    }


def deep_import(session, params, emit=None):
    """导入深拆结果：更新场景字段并把 shot_cards 存进叙事层。"""
    _require(session)
    text = params.get("text") or (Path(params["path"]).read_text(encoding="utf-8")
                                  if params.get("path") else "")
    if not text.strip():
        raise ValueError("缺少要导入的 JSON（text 或 path）")
    payload = unwrap_payload(extract_json(text))
    if not isinstance(payload, dict):
        raise ValueError("返回的 JSON 顶层不是对象，无法导入")
    narrative = data(session)
    scene = _pick_scene(narrative, payload.get("scene_id"), params.get("scene_index"))
    numbers = {row["number"] + 1 for row in session.scoped_shots({})}
    cards, warnings = [], []
    card_items, junk_cards = _dict_items(_first(payload, "shot_cards", "shots"))
    if junk_cards:
        warnings.append(f"shot_cards 跳过 {junk_cards} 个非对象项")
    for item in card_items:
        number = item.get("shot")
        if not isinstance(number, int) or number not in numbers:
            warnings.append(f"镜头卡 shot={number} 不在镜头表内，已丢弃")
            continue
        cards.append({"shot": number,
                      "action": str(item.get("action", ""))[:1000],
                      "dialogue": str(item.get("dialogue", "") or "")[:1000],
                      "technique": str(item.get("technique", "") or "")[:500]})
    cards.sort(key=lambda c: c["shot"])
    updated = deepcopy(scene)
    for key in ("title", "type", "function", "techniques", "key_beats"):
        if payload.get(key):
            updated[key] = str(payload[key])[:3000]
    updated["shot_cards"] = cards
    updated["deep_prompt_version"] = PROMPT_VERSION
    scenes = [updated if s["id"] == scene["id"] else s for s in narrative["scenes"]]
    session._annotate("导入场景深拆",
                      lambda: setattr(session, "narrative", dict(narrative, scenes=scenes)))
    return {"imported": True, "scene_id": scene["id"], "shot_cards": len(cards),
            "warnings": warnings}


# -- API 直连（P2；BYOK，密钥只在本机 ~/.cadensview/vlm.json 或环境变量） --------

def run_direct(session, params, cancel=None, progress=None):
    """构建单条多模态消息直连 OpenAI 兼容端点，返回原文并尝试自动导入。

    params: base_url / model / api_key（或 api_key_env）/ mode / max_images / timeout_s /
    no_import。默认取 ~/.cadensview/vlm.json 的 openai_compatible 配置（vlm-config 管理）。
    """
    _require(session)
    import base64
    import os

    from shotlab.vlm import chat_content, load_config, request_chat
    config = load_config()
    base_url = str(params.get("base_url") or config.get("base_url") or "").rstrip("/")
    model = str(params.get("model") or config.get("model") or "")
    api_key = str(params.get("api_key") or "")
    if not api_key and params.get("api_key_env"):
        api_key = os.environ.get(str(params["api_key_env"]), "")
    if not api_key:
        api_key = config.get("api_key", "")
    if not (base_url and model and api_key):
        raise ValueError("直连需要 base_url、model 与 api_key（或先 vlm-config 配置 openai_compatible）")

    manifest = build_manifest(session, params)
    rows = session.scoped_shots(params)
    sampled, _ = _sample_rows(rows, int(params.get("max_images", DEFAULT_MAX_IMAGES)))
    labels = {row["number"]: _shot_label(row, session) for row in rows}
    thumbs = _decode_thumbs(session, sampled, SHEET_TILE_W, cancel, progress)

    text_part = build_prompt(manifest) + "\n\n镜头表（manifest）：\n" + json.dumps(manifest, ensure_ascii=False)
    if session.subtitle_cues:
        text_part += "\n\n字幕（SRT）：\n" + build_srt(session)
    parts = [{"type": "text", "text": text_part}]
    for array, row in thumbs:
        buffer = BytesIO()
        _tile_image(array, SHEET_TILE_W, SHEET_TILE_H, labels[row["number"]]).save(
            buffer, "JPEG", quality=SHEET_JPEG_QUALITY)
        parts.append({"type": "image_url",
                      "image_url": {"url": "data:image/jpeg;base64,"
                                         + base64.b64encode(buffer.getvalue()).decode()}})

    try:
        timeout = int(params.get("timeout_s", 600))
    except (TypeError, ValueError) as error:
        raise ValueError("timeout_s 须为正整数秒") from error
    if timeout <= 0:
        raise ValueError("timeout_s 须为正整数秒")
    answer = request_chat(
        base_url + "/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": parts}]},
        {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}, timeout)
    content, served = chat_content(answer)
    if not content.strip():
        # An empty reply (refusal, content filter, reasoning-only model) must not be imported.
        raise ValueError("模型返回空内容（没有 choices 或 content），请重试或更换模型")
    result = {"model": served or model, "raw_chars": len(content)}
    if params.get("no_import"):
        return dict(result, raw=content)
    # The import has to see the same shot scope the model was given, or time-anchored
    # scenes would be mapped onto the wrong shots.
    scope = {key: params[key] for key in ("shot_ids", "exclude_regions") if key in params}
    imported = import_analysis(session, {"text": content, "model": served or model, **scope})
    return dict(result, imported=imported["imported"], preview=imported["preview"])


# -- 拉片长图（P3） --------------------------------------------------------------

def _wrap(text, limit):
    return [text[i:i + limit] for i in range(0, len(text), limit)] if text else []


def draw_sheet(session, params, emit=None):
    """拉片长图 PNG：情绪曲线 + 剧情线泳道 + 场景表。"""
    _require(session)
    from PIL import Image, ImageDraw

    from shotlab.fonts import ui_font
    narrative = data(session)
    if not narrative["scenes"]:
        raise ValueError("叙事层为空，请先导入 AI 拉片结果")
    identity = narrative.get("identity") or build_identity(session)
    duration = max(float(identity.get("duration_s") or 1.0), 1.0)
    width = max(900, int(params.get("width", 1600)))
    margin = 24
    title_font, text_font, small_font = ui_font(20), ui_font(13), ui_font(11)

    scenes, lines = narrative["scenes"], narrative["story_lines"]
    lane_index = {line["id"]: i for i, line in enumerate(lines)}

    macro_lines = _wrap(narrative["macro"].get("overall_structure", ""), 64)[:2]
    header_h = 58
    macro_h = 26 * len(macro_lines) + 12
    curve_h = 236
    lane_h = 30
    lanes_h = lane_h * len(lines) + 30
    row_h = 26
    height = header_h + macro_h + curve_h + lanes_h + row_h * len(scenes) + 66
    image = Image.new("RGB", (width, height), (18, 18, 24))
    draw = ImageDraw.Draw(image)
    plot_left, plot_w = margin + 70, width - margin * 2 - 70

    def x(seconds):
        return plot_left + min(max(seconds / duration, 0.0), 1.0) * plot_w

    draw.text((margin, 16), f"{identity.get('title', '')} · AI 拉片",
              fill=(240, 240, 240), font=title_font)
    meta = (f"{len(scenes)} 场景 · {len(narrative['curve_points'])} 曲线点"
            + (f" · {narrative.get('model')}" if narrative.get("model") else ""))
    draw.text((width - margin - 430, 24), meta, fill=(150, 150, 160), font=text_font)

    y = header_h
    for line in macro_lines:
        draw.text((margin, y), line, fill=(170, 180, 195), font=text_font)
        y += 26

    # 情绪曲线
    draw.text((margin, y), "观众投入强度（intensity 0-100）与情绪方向（valence ±）",
              fill=(210, 210, 220), font=text_font)
    top, baseline = y + 30, y + 30 + 170
    for i in range(0, 101, 25):
        yy = baseline - i * 1.7
        draw.line([(margin, yy), (width - margin, yy)], fill=(38, 38, 48))
        draw.text((margin + 4, yy - 7), str(i), fill=(110, 110, 125), font=small_font)
    draw.line([(margin, baseline), (width - margin, baseline)], fill=(70, 70, 84))
    points = narrative["curve_points"]
    for point in points:
        px = x(point["time"])
        if point["valence"]:
            half = min(abs(point["valence"]) / 100, 1) * 13
            color = (90, 200, 120) if point["valence"] > 0 else (230, 100, 90)
            draw.line([(px - half, baseline + 6), (px + half, baseline + 6)], fill=color, width=4)
    if len(points) > 1:
        draw.line([(x(p["time"]), baseline - (p["intensity"] or 0) * 1.7) for p in points],
                  fill=(90, 160, 255), width=2)
    for point in points:
        px, py = x(point["time"]), baseline - (point["intensity"] or 0) * 1.7
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(90, 160, 255))
        if point.get("rhythm_role") in ("peak", "low"):
            draw.text((px + 5, py - 16), point["title"][:16], fill=(230, 220, 150), font=small_font)
    for scene in scenes:
        draw.line([(x(scene["start_s"]), top), (x(scene["start_s"]), baseline + 12)],
                  fill=(52, 52, 64))

    # 剧情线泳道
    lanes_top = baseline + 34
    draw.text((margin, lanes_top - 22), "剧情线泳道（场景块按 primary_line 归道）",
              fill=(210, 210, 220), font=text_font)
    for i, line in enumerate(lines):
        lane_y = lanes_top + i * lane_h
        draw.text((margin, lane_y + 6), line["title"][:10], fill=(160, 160, 172), font=small_font)
        draw.line([(plot_left, lane_y + lane_h - 4), (width - margin, lane_y + lane_h - 4)],
                  fill=(40, 40, 50))
    for scene in scenes:
        lane = lane_index.get(scene.get("primary_line") or "", 0)
        lane_y = lanes_top + lane * lane_h
        x0, x1 = x(scene["start_s"]), x(scene["end_s"])
        index = SCENE_TYPES.index(scene["type"]) if scene["type"] in SCENE_TYPES else 7
        color = SCENE_PALETTE[index % len(SCENE_PALETTE)]
        fill = tuple(int(c * 0.45) for c in color)
        draw.rectangle([x0, lane_y + 3, max(x1 - 2, x0 + 8), lane_y + lane_h - 7],
                       fill=fill, outline=color)
        label = f"S{scene['start_shot']}-{scene['end_shot']} {scene['title']}"
        if x1 - x0 > 70:
            draw.text((x0 + 4, lane_y + 6), label[:int((x1 - x0) / 12)],
                      fill=(235, 235, 240), font=small_font)
        for shared in scene.get("shared_lines") or []:
            other = lane_index.get(shared)
            if other is not None and other != lane:
                oy = lanes_top + other * lane_h
                draw.rectangle([x0, oy + 3, max(x1 - 2, x0 + 8), oy + lane_h - 7],
                               outline=color, width=1)

    # 场景表
    table_top = lanes_top + lanes_h
    draw.text((margin, table_top), "场景表", fill=(210, 210, 220), font=text_font)
    columns = (0, 44, 128, 190, 250, 570)
    heads = ("#", "镜头", "时间", "类型", "标题", "作用")
    for value, column in zip(heads, columns):
        draw.text((margin + column, table_top + 24), value, fill=(130, 130, 145), font=small_font)
    for index, scene in enumerate(scenes, 1):
        y = table_top + 46 + (index - 1) * row_h
        cells = (str(index), f"S{scene['start_shot']}-{scene['end_shot']}",
                 timecode(scene["start_s"]), scene["type"][:4],
                 scene["title"][:24], scene["function"][:66])
        for value, column in zip(cells, columns):
            draw.text((margin + column, y), value, fill=(200, 200, 210), font=small_font)

    draw.text((margin, height - 22),
              f"{identity.get('source', '')} · {narrative.get('updated_at', '')}",
              fill=(110, 110, 125), font=small_font)
    path = Path(params["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path), "PNG")
    return {"path": str(path.resolve()), "scenes": len(scenes),
            "curve_points": len(points), "size": image.size}


# -- 跨片叙事比较（P3；只读其他项目文件，不写回） --------------------------------

def _narrative_of_project(path):
    """读取另一个 .shotlab 的叙事层与镜头数；没有叙事层时给空结果并说明。"""
    from shotlab.storage import ProjectStore
    store = ProjectStore(path)
    try:
        project, _index, settings = store.load()
    finally:
        store.close()
    metadata = settings.get("metadata", {})
    label = metadata.get("title") or Path(path).stem
    if metadata.get("episode"):
        label = f"{label} #{metadata['episode']}"
    narrative = settings.get("narrative") or {}
    if not isinstance(narrative, dict):
        narrative = {}
    raw_scenes = narrative.get("scenes") or []
    scenes = [s for s in (raw_scenes if isinstance(raw_scenes, list) else [])
              if isinstance(s, dict) and isinstance(s.get("start_shot"), int) and isinstance(s.get("end_shot"), int)]
    return label, project.completed_count, narrative, scenes


def compare_narratives(paths):
    """多片叙事层比较：场景数/场长（镜头数与占比）、类型分布、曲线特征。

    单个项目读不出来只降级为 error 行，不拖垮整张表；重复路径去重。
    """
    unique, seen = [], set()
    for path in paths:
        resolved = str(Path(path).resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    films = []
    for path in unique:
        try:
            label, shot_count, narrative, scenes = _narrative_of_project(path)
        except (ValueError, RuntimeError, OSError, sqlite3.Error) as error:
            films.append({"path": str(Path(path).resolve()), "error": str(error)[:300]})
            continue
        lengths = [max(s["end_shot"] - s["start_shot"] + 1, 1) for s in scenes]
        types = {}
        for scene in scenes:
            kind = scene["type"] if isinstance(scene.get("type"), str) else ""
            types[kind or "未分类"] = types.get(kind or "未分类", 0) + 1
        # 另一个项目的叙事层可能被手工改过：Python 的 json.loads 收得下 NaN/Infinity 字面量，
        # 只有有限数值才进入均值。
        raw_points = narrative.get("curve_points") or []
        points = [p for p in (raw_points if isinstance(raw_points, list) else [])
                  if isinstance(p, dict) and _finite(p.get("time"))]
        intensities = [p["intensity"] for p in points if _finite(p.get("intensity"))]
        valences = [p["valence"] for p in points if _finite(p.get("valence"))]
        macro = narrative.get("macro")
        structure = macro.get("overall_structure", "") if isinstance(macro, dict) else ""
        films.append({
            "path": str(Path(path).resolve()), "label": label,
            "has_narrative": bool(scenes or narrative.get("updated_at")),
            "shots": shot_count, "scenes": len(scenes),
            "median_scene_shots": _median(lengths),
            "scene_share": round(len(scenes) / shot_count, 4) if shot_count else None,
            "type_distribution": dict(sorted(types.items(), key=lambda kv: -kv[1])),
            "curve_points": len(points),
            "peaks": sum(1 for p in points if p.get("rhythm_role") == "peak"),
            "lows": sum(1 for p in points if p.get("rhythm_role") == "low"),
            "mean_intensity": round(sum(intensities) / len(intensities), 1) if intensities else None,
            "mean_valence": round(sum(valences) / len(valences), 1) if valences else None,
            "macro_structure": structure if isinstance(structure, str) else str(structure),
        })
    return {"films": films,
            "notes": ["场景场长按镜头数计（叙事层锚定单位）；曲线统计来自导入时的 emotion curve",
                      "has_narrative=false 的项目需先跑全片拉片再比较"]}


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return round((ordered[mid - 1] + ordered[mid]) / 2, 2)


def _finite(value):
    """True for a real, finite number; bools and the NaN/Infinity of hand-edited JSON are not."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def narrative_compare(session, params, emit=None):
    """比较当前项目与其他 .shotlab 的叙事层（params: others=[路径]）。"""
    _require(session)
    if not session.project_path:
        raise ValueError("当前项目尚未保存为文件")
    paths = [session.project_path, *(params.get("others") or [])]
    return compare_narratives(paths)


# -- 注册 -----------------------------------------------------------------------

# 同步 RPC 的第三个位置参数是 emit；长耗时函数走 job 时才拿 cancel/progress。
def _package_sync(session, params, emit=None):
    return export_package(session, params)


def _deep_sync(session, params, emit=None):
    return export_deep(session, params)


def _run_sync(session, params, emit=None):
    return run_direct(session, params)


def _package_job(session, params, cancel, progress):
    return export_package(session, params, cancel=cancel, progress=progress)


def _deep_job(session, params, cancel, progress):
    return export_deep(session, params, cancel=cancel, progress=progress)


def _run_job(session, params, cancel, progress):
    return run_direct(session, params, cancel=cancel, progress=progress)


def register(methods, jobs):
    methods.update({
        "narrative.get": narrative_get,
        "narrative.compare": narrative_compare,
        "lapian.package": _package_sync,
        "lapian.import": import_analysis,
        "lapian.deep": _deep_sync,
        "lapian.deep_import": deep_import,
        "lapian.run": _run_sync,
        "export.lapian_sheet": draw_sheet,
    })
    jobs.update({
        "lapian.package": ("export", _package_job),
        "lapian.deep": ("export", _deep_job),
        "lapian.run": ("vlm", _run_job),
    })
