"""Vision-language model annotation of Shots (opt-in; frames leave the machine).

Pixel statistics cannot tell a close-up from a long shot in a drawn image, a
camera tilt from a cel slide, or an eyecatch from a title card. A vision model
can propose those labels from a few keyframes per Shot. Results are stored as a
separate, labelled annotation layer ("VLM 推断"), never merged into manual labels,
and can be scored against manual labels with Cohen's kappa.

Backends:
- "anthropic": Claude through the official `anthropic` SDK (structured JSON
  output; server-side refusal fallback enabled).
- "openai_compatible": any /chat/completions endpoint that accepts image_url
  data URIs (e.g. Qwen-VL via DashScope compatible mode, GLM-4V, Ollama, LM Studio).

Configuration lives in ~/.cadensview/vlm.json or CADENSVIEW_VLM_* variables;
API keys are never written into project files.
"""

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from shotlab.statistics import AnalysisScope, build_context
from t0.media import FrameReader

PROMPT_VERSION = 1
CONFIG_PATH = Path.home() / ".cadensview" / "vlm.json"
DEFAULTS = dict(provider="anthropic", model="claude-opus-5-5", base_url="", effort="low",
                concurrency=2, frames=3, image_width=512, timeout_s=120)
SCALES = ["ECS", "CS", "MS", "FS", "LS"]
CAMERA = ["static", "pan", "tilt", "track", "zoom", "shake", "complex"]
ANGLES = ["eye", "high", "low", "overhead", "dutch"]
SETTINGS = ["interior", "exterior", "abstract", "unknown"]
CONTENT = ["dialogue", "reaction", "action", "establishing", "insert", "montage", "title_card", "text", "other"]
SEGMENTS = ["none", "op", "ed", "eyecatch", "preview", "recap", "title"]
EFFECTS = ["glow", "lens_flare", "blur", "light_rays", "speed_lines", "particles", "rain_snow", "none"]

SCHEMA = {
    "type": "object",
    "properties": {
        "shot_scale": {"type": "string", "enum": SCALES},
        "camera_movement": {"type": "string", "enum": CAMERA},
        "camera_angle": {"type": "string", "enum": ANGLES},
        "subjects": {"type": "integer"},
        "setting": {"type": "string", "enum": SETTINGS},
        "content_type": {"type": "string", "enum": CONTENT},
        "special_segment": {"type": "string", "enum": SEGMENTS},
        "effects": {"type": "array", "items": {"type": "string", "enum": EFFECTS}},
        "description": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["shot_scale", "camera_movement", "camera_angle", "subjects", "setting", "content_type",
                 "special_segment", "effects", "description", "confidence"],
    "additionalProperties": False,
}

SYSTEM = """You annotate single shots from films and anime for quantitative film-style research.
You receive keyframes from ONE shot, in temporal order. Judge the shot as a whole.

shot_scale (largest framed human/character subject; for no subject, judge by the main object):
  ECS extreme close-up (eyes, mouth, hand detail), CS close-up (head and shoulders),
  MS medium shot (waist up), FS full shot (whole body), LS long shot (figures small in the setting).
camera_movement across the keyframes: static, pan (horizontal), tilt (vertical), track (camera or
  layers travel, incl. anime cel/background slides), zoom (incl. anime T.U./T.B.), shake, complex.
camera_angle: eye, high, low, overhead, dutch.
subjects: number of distinct characters visible.
content_type: dialogue, reaction, action, establishing, insert, montage, title_card, text, other.
special_segment: op (opening credits sequence), ed (ending credits), eyecatch (anime mid-episode
  card), preview (next-episode preview), recap, title (title card), or none.
effects: compositing effects visible (glow, lens_flare, blur, light_rays, speed_lines, particles,
  rain_snow) or ["none"].
description: one short sentence in Simplified Chinese describing what happens.
confidence: 0-1, how sure you are of shot_scale and camera_movement.
If frames are dark or ambiguous, still choose the closest label and lower confidence.
Return only the JSON object."""


def load_config():
    config = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            config.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    for key in ("provider", "model", "base_url", "api_key", "effort"):
        value = os.environ.get(f"CADENSVIEW_VLM_{key.upper()}")
        if value:
            config[key] = value
    return config


def public_config(config=None):
    config = load_config() if config is None else config
    has_key = bool(config.get("api_key")) or (
        config.get("provider") == "anthropic" and bool(os.environ.get("ANTHROPIC_API_KEY")))
    return dict({k: v for k, v in config.items() if k != "api_key"}, has_key=has_key,
                config_path=str(CONFIG_PATH))


def save_config(values):
    config = load_config() if CONFIG_PATH.exists() else dict(DEFAULTS)
    for key in ("provider", "model", "base_url", "effort"):
        if key in values and values[key] is not None:
            config[key] = str(values[key]).strip()
    for key in ("concurrency", "frames", "image_width", "timeout_s"):
        if key in values and values[key] is not None:
            config[key] = int(values[key])
    if values.get("api_key"):
        config["api_key"] = str(values["api_key"]).strip()
    if values.get("clear_key"):
        config.pop("api_key", None)
    if config["provider"] not in ("anthropic", "openai_compatible"):
        raise ValueError("provider 只能是 anthropic 或 openai_compatible")
    if config["provider"] == "openai_compatible" and not config.get("base_url"):
        raise ValueError("openai_compatible 需要 base_url，例如 https://dashscope.aliyuncs.com/compatible-mode/v1")
    config["concurrency"] = max(1, min(config["concurrency"], 8))
    config["frames"] = max(1, min(config["frames"], 6))
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return public_config(config)


def _jpeg(frame, width):
    from PIL import Image
    image = Image.fromarray(frame.to_ndarray(format="rgb24"), "RGB")
    if image.width > width:
        image = image.resize((width, max(1, round(image.height * width / image.width))), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=82)
    return base64.standard_b64encode(buffer.getvalue()).decode("ascii")


def keyframes(reader, shot, count, width):
    span = shot.end_frame_exclusive - shot.start_frame
    picks = sorted({shot.start_frame + min(span - 1, int(span * (i + 0.5) / count)) for i in range(count)})
    return [_jpeg(reader.read(number), width) for number in picks]


def parse_reply(text):
    """JSON object from a model reply; tolerates code fences and leading prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型没有返回 JSON")
    return normalize(json.loads(text[start:end + 1]))


def normalize(data):
    """Keep only schema fields with valid values; unknown enums become None."""
    def pick(key, allowed):
        value = str(data.get(key, "")).strip()
        return value if value in allowed else None
    effects = [e for e in (data.get("effects") or []) if e in EFFECTS and e != "none"]
    try:
        subjects = max(0, int(data.get("subjects", 0)))
    except (TypeError, ValueError):
        subjects = None
    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = None
    return dict(shot_scale=pick("shot_scale", SCALES), camera_movement=pick("camera_movement", CAMERA),
                camera_angle=pick("camera_angle", ANGLES), subjects=subjects, setting=pick("setting", SETTINGS),
                content_type=pick("content_type", CONTENT), special_segment=pick("special_segment", SEGMENTS),
                effects=effects, description=str(data.get("description", ""))[:300], confidence=confidence)


def _user_text(shot, count):
    return (f"Shot #{shot.shot_number}: {float(shot.duration):.2f} s, "
            f"{shot.end_frame_exclusive - shot.start_frame} frames. {count} keyframes in order follow.")


class AnthropicBackend:
    def __init__(self, config):
        import anthropic
        self.anthropic = anthropic
        key = config.get("api_key")
        self.client = anthropic.Anthropic(api_key=key, timeout=config["timeout_s"]) if key \
            else anthropic.Anthropic(timeout=config["timeout_s"])
        self.model, self.effort = config["model"], config.get("effort") or "low"

    def annotate(self, shot, images):
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}
                   for data in images]
        content.append({"type": "text", "text": _user_text(shot, len(images))})
        response = self.client.beta.messages.create(
            model=self.model, max_tokens=4096, system=SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        if response.stop_reason == "refusal":
            raise RuntimeError("模型拒绝了此镜头")
        text = next((block.text for block in response.content if block.type == "text"), "")
        return parse_reply(text), response.model

    def retryable(self, error):
        return isinstance(error, (self.anthropic.RateLimitError, self.anthropic.APIConnectionError,
                                  self.anthropic.InternalServerError))


class OpenAICompatibleBackend:
    def __init__(self, config):
        self.url = config["base_url"].rstrip("/") + "/chat/completions"
        self.key = config.get("api_key") or ""
        self.model, self.timeout = config["model"], config["timeout_s"]

    def annotate(self, shot, images):
        content = [{"type": "text", "text": _user_text(shot, len(images))}]
        content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}} for data in images]
        body = json.dumps({"model": self.model, "messages": [
            {"role": "system", "content": SYSTEM + "\nJSON schema: " + json.dumps(SCHEMA)},
            {"role": "user", "content": content}]}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as reply:
            payload = json.loads(reply.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"]
        if isinstance(text, list):
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        return parse_reply(text), payload.get("model", self.model)

    def retryable(self, error):
        return (isinstance(error, urllib.error.HTTPError) and error.code in (408, 429, 500, 502, 503, 504)) \
            or isinstance(error, (urllib.error.URLError, TimeoutError))


def backend_for(config):
    if config["provider"] == "anthropic":
        try:
            return AnthropicBackend(config)
        except ImportError as error:
            raise ValueError("未安装 anthropic SDK：uv pip install anthropic") from error
    return OpenAICompatibleBackend(config)


def _call(backend, shot, images, attempts=3):
    for attempt in range(attempts):
        try:
            return backend.annotate(shot, images)
        except Exception as error:  # noqa: BLE001 — 只有可重试错误才重试
            if attempt + 1 == attempts or not backend.retryable(error):
                raise
            time.sleep(2 ** attempt)


def annotate_shots(project, index, scope=AnalysisScope(), existing=None, force=False,
                   config=None, backend=None, cancel=None, progress=None):
    """Annotate completed Shots in scope; returns ({shot_id: label}, errors, meta)."""
    config = load_config() if config is None else config
    backend = backend_for(config) if backend is None else backend
    existing = existing or {}
    records = build_context(project, index, scope).records
    todo = [s for s in records if force or not (
        existing.get(s.shot_id, {}).get("start") == s.start_frame
        and existing.get(s.shot_id, {}).get("end") == s.end_frame_exclusive
        and existing.get(s.shot_id, {}).get("model") == config["model"]
        and existing.get(s.shot_id, {}).get("prompt_version") == PROMPT_VERSION)]
    results, errors = {}, []
    if not todo:
        return results, errors, dict(requested=0)
    reader = FrameReader(index, sequential=False)
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
            futures = {}
            for shot in todo:
                if cancel is not None and cancel.is_set():
                    break
                images = keyframes(reader, shot, config["frames"], config["image_width"])
                futures[pool.submit(_call, backend, shot, images)] = shot
                # 保持在途请求有限，取消时不至于还有一大批已发出。
                while len([f for f in futures if not f.done()]) >= config["concurrency"] * 2:
                    time.sleep(0.05)
            for future in as_completed(futures):
                if cancel is not None and cancel.is_set():
                    for pending in futures:
                        pending.cancel()
                    break
                shot = futures[future]
                done += 1
                try:
                    label, model = future.result()
                    results[shot.shot_id] = dict(
                        label, start=shot.start_frame, end=shot.end_frame_exclusive, model=config["model"],
                        served_by=model, provider=config["provider"], prompt_version=PROMPT_VERSION,
                        at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                except Exception as error:  # noqa: BLE001 — 单镜头失败不影响其余
                    errors.append(dict(shot_id=shot.shot_id, shot_number=shot.shot_number,
                                       error=f"{type(error).__name__}: {error}"[:300]))
                if progress is not None:
                    progress(done, len(todo))
    finally:
        reader.close()
    if cancel is not None and cancel.is_set() and not results:
        raise CancelledError("VLM 标注已取消")
    meta = dict(provider=config["provider"], model=config["model"], prompt_version=PROMPT_VERSION,
                frames_per_shot=config["frames"], image_width=config["image_width"], requested=len(todo))
    return results, errors, meta


def region_suggestions(rows):
    """Contiguous Shots the VLM tagged as OP/ED/preview/... → candidate content regions."""
    suggestions = []
    for row in rows:
        segment = (row.get("vlm") or {}).get("special_segment")
        if not segment or segment == "none" or row.get("pending"):
            continue
        if suggestions and suggestions[-1]["label"] == segment and suggestions[-1]["end_frame"] == row["start"]:
            suggestions[-1].update(end_frame=row["end"], shots=suggestions[-1]["shots"] + 1)
        else:
            suggestions.append(dict(label=segment, start_frame=row["start"], end_frame=row["end"], shots=1))
    labels = {"op": "OP", "ed": "ED", "eyecatch": "eyecatch", "preview": "preview", "recap": "recap",
              "title": "title"}
    return [dict(s, label=labels.get(s["label"], "other")) for s in suggestions
            if s["shots"] >= 2 or s["label"] in ("eyecatch", "title")]
