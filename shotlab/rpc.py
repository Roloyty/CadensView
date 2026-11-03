"""JSON-RPC sidecar for native GUI clients; one request per line on stdin.

Short requests answer synchronously: {"id", "method", "params"} in, a matching
{"id", "ok", "result"|"error"} out. Long jobs run off-thread through job.start /
job.cancel and report {"event": "job", "job_id", "phase": progress|done|error}.
While a job runs the session is busy and rejects state-mutating requests so the
job never races an edit. All state (index, project, history, store) lives here so
the domain rules, undo and storage stay inside the tested core; the GUI is a view.
"""

import json
import sys
import threading
from pathlib import Path

import numpy as np

from .domain import Project, Shot
from .history import History
from .storage import ProjectStore
from t0.media import FrameIndex, build_index

# 任务运行期间仍允许的请求（只读/控制类）。
BUSY_ALLOWED = {"ping", "project.state", "job.cancel"}


def _session_store(path):
    """Keep core validation; busy gating serializes this session's SQLite access.

    A successful index.build may retire the store on its job thread. Use a
    thread-shareable connection so failed indexing can keep the old store open.
    """
    import sqlite3
    store = ProjectStore(path)
    try:
        connection = sqlite3.connect(store.path.as_uri() + "?mode=rw", uri=True,
                                     timeout=2, check_same_thread=False)
    finally:
        store.close()
    store.connection = connection
    return store


def _same_path(first, second):
    first, second = Path(first).resolve(), Path(second).resolve()
    return first == second or (first.exists() and second.exists() and first.samefile(second))


class Session:
    """One open video/project; the sidecar serves a single session at a time."""

    def __init__(self):
        self.video_path = None
        self.index = None
        self.project = None
        self.history = History()
        self.store = None
        self.settings = {}
        self.project_path = None
        self.dirty = False
        self.stats = None  # 最近一次 shot_statistics 结果缓存
        self.busy = None   # 正在运行的后台任务名（index.build / analysis.run）
        self.shot_tags = {}  # {shot_id: [tag1, tag2, ...]}（存 settings，零迁移）
        self.scale_labels = {}  # {shot_id: {label, label_zh, confidence}}（存 settings）
        self.subtitle_cues = []  # 外挂 SRT 字幕 [{start_us, end_us, text}]
        self.motion = None  # 最近一次运动分析结果
        self.motion_key = None

    def start_busy(self, name):
        if self.busy:
            raise ValueError(f"已有后台任务运行中（{self.busy}），请等待或取消")
        self.busy = name

    def end_busy(self):
        self.busy = None

    def close_current_store(self):
        if self.store is not None:
            self.store.close()
            self.store = None

    def _reset_metadata(self, settings=None):
        self.settings = settings if settings is not None else {}
        self.shot_tags = self.settings.get("shot_tags", {})
        self.scale_labels = self.settings.get("scale_labels", {})
        self.subtitle_cues = []
        self.motion = None
        self.motion_key = None
        self.history = History()
        self.stats = None

    def analysis_scope(self, params):
        """Omission selects completed coverage; explicit IDs never widen it."""
        from shotlab.statistics import AnalysisScope
        if self.project is None or self.index is None:
            raise ValueError("先打开视频并建立索引")
        if "shot_ids" not in params:
            return AnalysisScope()
        ids = params["shot_ids"]
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids):
            raise ValueError("shot_ids 必须是非空的已完成镜头 ID 列表")
        completed = self.project.shots[:self.project.completed_count]
        selected = set(ids)
        if selected - {shot.id for shot in completed}:
            raise ValueError("shot_ids 包含未知或未完成镜头 ID")
        return AnalysisScope(kind="selected_shots",
                             shot_ids=tuple(shot.id for shot in completed if shot.id in selected))

    def scoped_shots(self, params):
        scope = self.analysis_scope(params)
        selected = set(scope.shot_ids) if scope.kind == "selected_shots" else None
        return [shot for shot in self.state()["shots"]
                if not shot["pending"] and (selected is None or shot["id"] in selected)]

    def motion_run(self, scope, cancel=None, progress=None):
        from shotlab.motion import analyze_motion
        result = analyze_motion(self.project, self.index, scope, cancel=cancel, progress=progress)
        self.motion = result
        self.motion_key = (self.project.id, scope.kind, tuple(scope.shot_ids))
        return result

    def motion_current(self, params):
        if self.motion is None or self.project is None:
            raise ValueError("先运行运动分析")
        scope = self.analysis_scope(params)
        if self.motion_key != (self.project.id, scope.kind, tuple(scope.shot_ids)):
            raise ValueError("范围或镜头边界已变化，请重新运行运动分析")
        return self.motion

    def stills_run(self, params):
        from shotlab.slit import export_stills
        scope = self.analysis_scope(params)
        return export_stills(self.project, self.index, params["path"],
                             count=int(params.get("count", 100)),
                             scope=scope, width=int(params.get("width", 480)))

    def slit_run(self, params, cancel=None, progress=None):
        from shotlab.slit import build_sheet
        from shotlab.statistics import build_context
        scope = self.analysis_scope(params)
        context = build_context(self.project, self.index, scope)
        return build_sheet(self.index, context.records, params["path"],
                           width=int(params.get("width", 160)),
                           columns=int(params.get("columns", 6)),
                           cancel=cancel, progress=progress)

    # -- snapshot -----------------------------------------------------------

    def state(self):
        project, index = self.project, self.index
        shots = []
        if project is not None and index is not None:
            completed = project.completed_count
            for number, shot in enumerate(project.shots):
                start, end = project.interval(number)
                try:
                    duration_s = ((index.pts[end - 1] + index.durations[end - 1])
                                  - index.pts[start]) * float(index.tb)
                except (IndexError, TypeError):
                    duration_s = 0.0
                scale = self.scale_labels.get(shot.id)
                if (not scale or number >= completed or
                        scale.get("start") != start or scale.get("end") != end):
                    scale = None
                shots.append({"id": shot.id, "number": number, "start": start, "end": end,
                              "note": shot.note, "pending": number >= completed,
                              "duration_s": round(duration_s, 4),
                              "tags": self.shot_tags.get(shot.id, []),
                              "scale": scale})
        return {
            "project_id": project.id if project else None,
            "video_path": self.video_path,
            "project_path": self.project_path,
            "dirty": self.dirty,
            "has_index": self.index is not None,
            "frame_count": project.frame_count if project else None,
            "current_frame": project.current_frame if project else 0,
            "selected_id": project.selected_id if project else "",
            "pending_start": project.pending_start if project else None,
            "completed_count": project.completed_count if project else 0,
            "can_undo": bool(self.history.undo),
            "can_redo": bool(self.history.redo),
            "shots": shots,
        }

    # -- lifecycle ----------------------------------------------------------

    def build_index(self, video_path, limit=None, emit=None, cancel=None):
        path = Path(video_path)
        if not path.is_file():
            raise ValueError(f"视频文件不存在: {video_path}")
        progress = (lambda count, seconds: emit("index", done=count, time_s=seconds)) if emit else None
        index = build_index(str(path), limit=limit, progress=progress, cancel=cancel)
        index.verify_source()
        self.close_current_store()
        self.index = index
        self.video_path = str(path.resolve())
        self.project = None
        self.project_path = None
        self._reset_metadata()
        self.dirty = False
        return {"index": self.index.__dict__, "video_path": self.video_path}

    def analysis_run(self, cancel=None, progress=None, *, scope=None):
        """颜色时间轴 + 镜头统计；仅由后台任务线程调用（busy 保护下无并发写）。"""
        from shotlab.statistics import build_context
        from shotlab.statistics.colors import analyze_colors
        from shotlab.statistics.shots import shot_statistics
        if self.project is None or self.index is None:
            raise ValueError("先打开视频并建立索引")
        if scope is None:
            scope = self.analysis_scope({})
        context = build_context(self.project, self.index, scope)
        result = analyze_colors(context, cancel=cancel, progress=progress, include_summary=True)
        self.stats = shot_statistics(context)
        return dict(result, shot_statistics=self.stats)

    def scale_run(self, cancel=None, progress=None):
        """逐镜头景别识别；仅由后台任务线程调用。"""
        from concurrent.futures import CancelledError
        from .scale import classify_shot_middle, LABELS_ZH
        if self.project is None or self.index is None:
            raise ValueError("先打开视频并建立索引")
        rows = [s for s in self.state()["shots"] if not s["pending"]]
        if not rows:
            raise ValueError("没有已完成镜头")
        results = {}
        for i, shot in enumerate(rows):
            if cancel is not None and cancel.is_set():
                raise CancelledError("景别识别已取消")
            label, confidence = classify_shot_middle(self.index, shot["start"], shot["end"])
            results[shot["id"]] = {
                "label": label,
                "label_zh": LABELS_ZH.get(label, label),
                "confidence": round(confidence, 3),
                "start": shot["start"], "end": shot["end"],
            }
            if progress is not None:
                progress(i + 1, len(rows))
        # 持久化：镜头边界变了自动失效（key 含 start/end）
        self.scale_labels = {k: v for k, v in results.items()}
        self.dirty = True
        return {"scales": results}

    def new_project(self):
        if self.index is None:
            raise ValueError("先建立视频索引")
        self.index.verify_source()
        shot = Shot()
        project = Project(len(self.index.pts), [0, len(self.index.pts)], [shot],
                          selected_id=shot.id, pending_start=0)
        project.validate()
        self.close_current_store()
        self.project = project
        self.project_path = None
        self._reset_metadata()
        self.dirty = True
        return self.state()

    def open_project(self, path):
        store = _session_store(path)
        try:
            project, index, settings = store.load()
            index.verify_source()
        except Exception:
            store.close()
            raise
        self.close_current_store()
        self.store, self.project, self.index = store, project, index
        self.video_path = index.path
        self.project_path = str(store.path)
        self._reset_metadata(settings)
        self.dirty = False
        return self.state()

    def save_as(self, path, overwrite=False):
        import os
        from tempfile import TemporaryDirectory
        if self.project is None or self.index is None:
            raise ValueError("没有可保存的项目")
        if type(overwrite) is not bool:
            raise ValueError("overwrite 必须为布尔值")
        target = Path(path).resolve()
        if _same_path(target, self.index.path):
            raise ValueError("不能将项目保存到源视频路径")
        if self.project_path is not None and _same_path(target, self.project_path):
            # 同路径始终走普通保存，保留核心的并发写入冲突检测。
            return self.save()
        if target.exists() and not overwrite:
            raise FileExistsError("目标文件已存在；覆盖需要 overwrite=true")
        settings = dict(self.settings, shot_tags=self.shot_tags, scale_labels=self.scale_labels)
        target.parent.mkdir(parents=True, exist_ok=True)
        # 临时目录与目标同文件系统；只有完整、可重开的 SQLite 才能发布。
        with TemporaryDirectory(prefix=".shotlab-save-", dir=target.parent) as directory:
            staged = Path(directory) / "snapshot.shotlab"
            if self.store is not None:
                self.store.backup(staged, self.project, settings)
            else:
                created = ProjectStore.create(staged, self.index, self.project, settings)
                created.close()
            snapshot = ProjectStore(staged)
            try:
                if snapshot.connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    raise ValueError("临时项目 SQLite 校验失败")
                project, index, saved_settings = snapshot.load()
                index.verify_source()
                if project != self.project or index != self.index:
                    raise ValueError("临时项目与当前内存状态不一致")
            finally:
                snapshot.close()
            if overwrite:
                os.replace(staged, target)
            else:
                # 原子且不覆盖：即便确认后有其他进程创建目标，也必须拒绝。
                os.link(staged, target)
        store = _session_store(target)
        self.close_current_store()
        self.store = store
        self.settings = saved_settings
        self.project_path = str(target)
        self.dirty = False
        return self.state()

    def save(self):
        if self.store is None:
            raise ValueError("项目尚未保存过文件")
        settings = dict(self.settings, shot_tags=self.shot_tags, scale_labels=self.scale_labels)
        self.store.save(self.project, settings)
        self.settings = settings
        self.dirty = False
        return self.state()

    def close_project(self):
        self.close_current_store()
        self.project = None
        self.project_path = None
        self._reset_metadata()
        self.dirty = False
        return self.state()

    # -- editing ------------------------------------------------------------

    def _edit(self, label, key=None, mutate=None, select_frame=None):
        """Apply one undoable edit; a raising mutate leaves state untouched."""
        import copy
        before = copy.deepcopy(self.project)
        if mutate is not None:
            mutate()
        if select_frame is not None:
            self.select(select_frame, record=False)
        self.project.validate()
        self.history.record(before, self.project, label, key)
        self.stats = None  # 镜头结构/笔记变化后统计缓存失效
        self.motion = None
        self.motion_key = None
        self.dirty = True
        return self.state()

    def select(self, frame=None, shot_id=None, record=False):
        # 光标/选中是易失的浏览状态，不算未保存改动。
        if shot_id:
            self.project.row(shot_id)  # ValueError when unknown
            if frame is None:
                frame = self.project.interval(self.project.row(shot_id))[0]
        if frame is not None:
            self.project.shot_at(frame)  # range check
            self.project.current_frame = int(frame)
        if shot_id:
            self.project.selected_id = shot_id
        return self.state()

    def apply_detection(self, result_dir):
        result = json.loads((Path(result_dir) / "result.json").read_text(encoding="utf-8"))
        cuts = sorted({int(b["frame"]) for b in result["boundaries"]})
        if self.project.pending_start is None:
            raise ValueError("分析已完成，不能覆盖已有镜头")
        ids = self.project.extend_analysis(cuts, complete=True)
        self.history.extend_analysis(cuts, complete=True, shot_ids=ids)
        self.stats = None
        self.dirty = True
        return self.state()


METHODS = {}

def method(name):
    def register(fn):
        METHODS[name] = fn
        return fn
    return register


# -- request handlers -------------------------------------------------------

@method("ping")
def _ping(session, params, emit=None):
    return {"version": "0.1.0"}


@method("index.build")
def _index_build(session, params, emit=None):
    return session.build_index(params["video_path"], limit=params.get("limit"), emit=emit)


@method("index.verify")
def _index_verify(session, params, emit=None):
    session.index.verify_source()
    return {"ok": True}


@method("index.get")
def _index_get(session, params, emit=None):
    if session.index is None:
        raise ValueError("尚未建立索引")
    return {"index": session.index.__dict__}


@method("project.new")
def _project_new(session, params, emit=None):
    return session.new_project()


@method("project.open")
def _project_open(session, params, emit=None):
    return session.open_project(params["path"])


@method("project.save")
def _project_save(session, params, emit=None):
    return session.save()


@method("project.save_as")
def _project_save_as(session, params, emit=None):
    return session.save_as(params["path"], overwrite=params.get("overwrite", False))


@method("project.close")
def _project_close(session, params, emit=None):
    return session.close_project()


@method("project.state")
def _project_state(session, params, emit=None):
    return session.state()


@method("edit.split")
def _edit_split(session, params, emit=None):
    return session._edit("拆分镜头", mutate=lambda: session.project.split(int(params["frame"])))


@method("edit.merge")
def _edit_merge(session, params, emit=None):
    return session._edit("合并镜头", mutate=lambda: session.project.merge(int(params["row"])))


@method("edit.move_boundary")
def _edit_move(session, params, emit=None):
    return session._edit("移动边界",
                         mutate=lambda: session.project.move_boundary(int(params["boundary"]), int(params["frame"])))


@method("edit.set_note")
def _edit_note(session, params, emit=None):
    shot_id, text = params["shot_id"], params.get("text", "")
    return session._edit("镜头笔记", key=f"note:{shot_id}",
                         mutate=lambda: session.project.set_note(shot_id, text))


@method("edit.select")
def _edit_select(session, params, emit=None):
    return session.select(frame=params.get("frame"), shot_id=params.get("shot_id"))


@method("edit.undo")
def _edit_undo(session, params, emit=None):
    session.project = session.history.restore(redo=False)
    session.project.validate()
    session.stats = None
    session.dirty = True
    return session.state()


@method("edit.redo")
def _edit_redo(session, params, emit=None):
    session.project = session.history.restore(redo=True)
    session.project.validate()
    session.stats = None
    session.dirty = True
    return session.state()


@method("project.apply_detection")
def _apply_detection(session, params, emit=None):
    return session.apply_detection(params["result_dir"])


@method("stats.run")
def _stats_run(session, params, emit=None):
    from shotlab.statistics import build_context
    from shotlab.statistics.shots import shot_statistics
    if session.project is None or session.index is None:
        raise ValueError("先打开视频并建立索引")
    context = build_context(session.project, session.index, session.analysis_scope(params))
    session.stats = shot_statistics(context)
    return session.stats


@method("palette.get")
def _palette_get(session, params, emit=None):
    from shotlab.statistics.colors import analyze_frame_palette
    if session.project is None or session.index is None:
        raise ValueError("先打开视频并建立索引")
    from shotlab.statistics import build_context
    context = build_context(session.project, session.index)
    frame = int(params["frame"])
    return analyze_frame_palette(context, frame)


@method("analysis.run")
def _analysis_run(session, params, emit=None):
    # 同步形态：测试与小批量用；GUI 走 job.start 以获得进度与取消。
    if session.project is None or session.index is None:
        raise ValueError("先打开视频并建立索引")
    return session.analysis_run(
        scope=session.analysis_scope(params),
        progress=(lambda done, total: emit("analysis", done=done, total=total)) if emit else None)


@method("export.shots_csv")
def _export_shots_csv(session, params, emit=None):
    """镜头汇总 CSV（utf-8-sig，Excel 直接打开中文不乱码）。"""
    import csv
    if session.project is None or session.index is None:
        raise ValueError("先打开视频并建立索引")
    path = Path(params["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = session.scoped_shots(params)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["镜头编号", "起始帧", "结束帧(不含)", "时长(秒)", "标签", "备注"])
        for shot in rows:
            writer.writerow([shot["number"] + 1, shot["start"], shot["end"],
                             f"{shot['duration_s']:.3f}",
                             "; ".join(shot.get("tags", [])),
                             shot["note"]])
    return {"written": len(rows), "path": str(path.resolve())}


@method("tags.set")
def _tags_set(session, params, emit=None):
    """替换镜头的全部标签。"""
    shot_id = params["shot_id"]
    tags = [t.strip() for t in params.get("tags", []) if t.strip()]
    session.project.row(shot_id)  # 校验镜头存在
    if tags:
        session.shot_tags[shot_id] = sorted(set(tags))
    else:
        session.shot_tags.pop(shot_id, None)
    session.dirty = True
    return session.state()


@method("tags.remove")
def _tags_remove(session, params, emit=None):
    """移除镜头的一个标签。"""
    shot_id, tag = params["shot_id"], params["tag"]
    tags = session.shot_tags.get(shot_id, [])
    if tag in tags:
        tags = [t for t in tags if t != tag]
        if tags:
            session.shot_tags[shot_id] = tags
        else:
            session.shot_tags.pop(shot_id, None)
        session.dirty = True
    return session.state()


@method("tags.vocab")
def _tags_vocab(session, params, emit=None):
    """全部标签词表（自动汇总）。"""
    vocab = set()
    for tags in session.shot_tags.values():
        vocab.update(tags)
    return {"vocabulary": sorted(vocab)}


@method("export.contact_sheet")
def _export_contact_sheet(session, params, emit=None):
    """镜头网格长图 PNG（Pillow 渲染编号+时长）。"""
    if session.project is None or session.index is None:
        raise ValueError("先打开视频并建立索引")
    from PIL import Image, ImageDraw, ImageFont
    from t0.media import FrameReader

    path = Path(params["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = int(params.get("columns", 5))
    thumb_w = int(params.get("thumb_width", 160))

    rows_data = session.scoped_shots(params)
    if not rows_data:
        raise ValueError("没有已完成镜头")

    reader = FrameReader(session.index)
    try:
        cell_h = 0
        thumbs = []
        for shot in rows_data:
            mid = min((shot["start"] + shot["end"]) // 2, len(session.index.pts) - 1)
            frame = reader.read(mid)
            arr = frame.to_ndarray(format="rgb24")
            h, w = arr.shape[:2]
            tw = thumb_w
            th = int(h * tw / w)
            if th == 0:
                continue
            # 最近邻缩放
            ys = (np.arange(th) * h // th).clip(0, h - 1)
            xs = (np.arange(tw) * w // tw).clip(0, w - 1)
            thumbs.append((arr[ys][:, xs], shot))
            cell_h = max(cell_h, th)
    finally:
        reader.close()

    import math
    total_rows = math.ceil(len(thumbs) / cols)
    label_h = 18
    sheet_w = cols * (thumb_w + 4) + 4
    sheet_h = total_rows * (cell_h + label_h + 4) + 4

    img = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 28))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 11)
    except OSError:
        font = ImageFont.load_default()

    for i, (thumb_arr, shot) in enumerate(thumbs):
        row, col = divmod(i, cols)
        x = col * (thumb_w + 4) + 4
        y = row * (cell_h + label_h + 4) + 4
        img.paste(Image.fromarray(thumb_arr), (x, y))
        label = f"#{shot['number']+1}  {shot['duration_s']:.1f}s"
        if shot.get("tags"):
            label += "  " + " ".join(shot["tags"])
        draw.text((x + 1, y + cell_h + 1), label, fill=(200, 200, 200), font=font)

    img.save(str(path), "PNG")
    return {"written": len(thumbs), "path": str(path.resolve())}


@method("export.gif_range")
def _export_gif_range(session, params, emit=None):
    """计算 GIF 导出所需的精确时间范围（帧号→秒）。"""
    if session.index is None:
        raise ValueError("先建立索引")
    start_frame = int(params["start_frame"])
    end_frame = int(params["end_frame"])
    start_frame = max(0, min(start_frame, len(session.index.pts) - 1))
    end_frame = max(start_frame + 1, min(end_frame, len(session.index.pts)))
    index = session.index
    start_s = float(index.pts[start_frame] * index.tb)
    end_s = float(index.pts[end_frame - 1] * index.tb) + float(index.durations[end_frame - 1] * index.tb)
    return {"start_s": start_s, "duration_s": end_s - start_s, "frame_count": end_frame - start_frame}


@method("motion.get")
def _motion_get(session, params, emit=None):
    return session.motion_current(params)


@method("export.motion_spectrum")
def _export_motion_spectrum(session, params, emit=None):
    from shotlab.motion import write_spectrum
    result = session.motion_current(params)
    return write_spectrum(result, params["path"], width=int(params.get("width", 1800)),
                          height=int(params.get("height", 300)))


@method("export.stills")
def _export_stills(session, params, emit=None):
    if session.project is None or session.index is None:
        raise ValueError("先打开视频并建立索引")
    return session.stills_run(params)


# -- protocol loop ----------------------------------------------------------

# 允许以后台任务方式运行的方法（长耗时、可取消）。
JOB_METHODS = {"index.build", "analysis.run", "scale.run", "motion.run", "slit.run"}


def serve(stdin=sys.stdin, stdout=sys.stdout, session=None):
    if hasattr(stdin, "reconfigure"):
        stdin.reconfigure(encoding="utf-8")
    if hasattr(stdout, "reconfigure"):
        stdout.reconfigure(encoding="utf-8")
    session = session if session is not None else Session()
    lock = threading.Lock()
    jobs = {}  # job_id -> {"cancel": Event, "name": str}
    counter = [0]

    def write(payload):
        with lock:
            _write(stdout, payload)

    def run_job(job_id, name, params):
        cancel = jobs[job_id]["cancel"]

        def emit(topic, **values):
            write({"event": "job", "job_id": job_id, "topic": topic, "phase": "progress", **values})

        result, failure = None, None
        try:
            if name == "index.build":
                result = session.build_index(params["video_path"], limit=params.get("limit"),
                                             emit=emit, cancel=cancel)
            elif name == "analysis.run":
                result = session.analysis_run(
                    scope=session.analysis_scope(params),
                    cancel=cancel,
                    progress=lambda done, total: emit("analysis", done=done, total=total))
            elif name == "scale.run":
                result = session.scale_run(
                    cancel=cancel,
                    progress=lambda done, total: emit("scale", done=done, total=total))
            elif name == "motion.run":
                result = session.motion_run(
                    session.analysis_scope(params), cancel=cancel,
                    progress=lambda done, total: emit("motion", done=done, total=total))
            elif name == "slit.run":
                result = session.slit_run(
                    params, cancel=cancel,
                    progress=lambda done, total: emit("slit", done=done, total=total))
            else:
                raise ValueError(f"不支持的任务: {name}")
        except Exception as error:
            failure = error
        finally:
            # 先解除忙态再发终态事件：GUI 收到 done 后立刻发起的后续请求不能被拒。
            jobs.pop(job_id, None)
            session.end_busy()
        if failure is None:
            write({"event": "job", "job_id": job_id, "topic": name, "phase": "done", "result": result})
        else:
            write({"event": "job", "job_id": job_id, "topic": name, "phase": "error",
                   "message": str(failure), "kind": type(failure).__name__})

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        request = {}
        try:
            request = json.loads(line)
            method = request.get("method")
            if method not in ("job.start", "job.cancel"):
                handler = METHODS[method]
            else:
                handler = None
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            write({"id": request.get("id") if isinstance(request, dict) else None,
                   "ok": False, "error": {"message": f"bad request: {error}"}})
            continue

        params = request.get("params") or {}

        if method == "job.start":
            name = params.get("method")
            if name not in JOB_METHODS:
                write({"id": request.get("id"), "ok": False,
                       "error": {"message": f"不支持的任务: {name}", "kind": "ValueError"}})
                continue
            try:
                session.start_busy(name)
            except ValueError as error:
                write({"id": request.get("id"), "ok": False,
                       "error": {"message": str(error), "kind": "Busy"}})
                continue
            with lock:
                counter[0] += 1
                job_id = counter[0]
            jobs[job_id] = {"cancel": threading.Event(), "name": name}
            # 先应答再启动：极快任务也不能让 done 事件抢在 job_id 之前到达。
            write({"id": request.get("id"), "ok": True, "result": {"job_id": job_id, "method": name}})
            threading.Thread(target=run_job, args=(job_id, name, params.get("params") or {}),
                             name=f"job-{job_id}", daemon=True).start()
            continue

        if method == "job.cancel":
            job = jobs.get(params.get("job_id"))
            if job is not None:
                job["cancel"].set()
            write({"id": request.get("id"), "ok": True, "result": {"cancelled": job is not None}})
            continue

        if session.busy and method not in BUSY_ALLOWED:
            write({"id": request.get("id"), "ok": False,
                   "error": {"message": f"后台任务运行中（{session.busy}），请稍等或取消",
                             "kind": "Busy"}})
            continue

        def emit(job, **values):
            write({"event": "progress", "job": job, **values})

        try:
            result = handler(session, params, emit)
            write({"id": request.get("id"), "ok": True, "result": result})
        except Exception as error:  # domain errors are part of the contract
            write({"id": request.get("id"), "ok": False,
                   "error": {"message": str(error), "kind": type(error).__name__}})


def _write(stdout, payload):
    stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stdout.flush()


@method("subtitles.load")
def _subtitles_load(session, params, emit=None):
    """加载外挂 SRT 字幕（原版 subtitles.pyc 移植）。"""
    import html
    from html.parser import HTMLParser

    class _PlainText(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []
        def handle_data(self, data):
            self.parts.append(data)
        def handle_starttag(self, tag, attrs):
            if tag == 'br':
                self.parts.append('\n')

    def plain_text(text):
        parser = _PlainText()
        parser.feed(text)
        return html.unescape(''.join(parser.parts)).strip()

    path = Path(params["path"])
    if not path.is_file():
        raise ValueError(f"字幕文件不存在: {params['path']}")

    def timestamp(value):
        value = value.strip().replace(',', '.')
        parts = value.split(':')
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        else:
            raise ValueError(f"bad timestamp: {value}")
        return round((int(h) * 3600 + int(m) * 60 + float(s)) * 1_000_000)

    cues = []
    current = []

    def flush():
        if not current:
            return
        try:
            index = next((i for i, l in enumerate(current) if '-->' in l), None)
            if index is not None:
                start_s, end_s = current[index].split('-->')
                text = plain_text('\n'.join(current[index + 1:]))
                if text:
                    cues.append({"start_us": timestamp(start_s),
                                 "end_us": timestamp(end_s),
                                 "text": text})
        except (ValueError, IndexError):
            pass
        current.clear()

    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line:
            flush()
        else:
            current.append(line)
    flush()  # 文件末尾没有空行时也要处理最后一条
    session.subtitle_cues = cues
    return {"cues": len(cues), "entries": cues, "path": str(path.resolve())}


@method("subtitles.clear")
def _subtitles_clear(session, params, emit=None):
    session.subtitle_cues = []
    return {"cleared": True}


@method("subtitles.at")
def _subtitles_at(session, params, emit=None):
    """获取指定时间（微秒）的字幕文本。"""
    us = int(params["timestamp_us"])
    for cue in getattr(session, 'subtitle_cues', []):
        if cue["start_us"] <= us < cue["end_us"]:
            return {"text": cue["text"]}
    return {"text": None}


@method("export.pdf_sheet")
def _export_pdf_sheet(session, params, emit=None):
    """全镜头 PDF 联系表（原版 contact_sheet.PdfSheet 移植，Pillow 渲染）。"""
    if session.project is None or session.index is None:
        raise ValueError("先打开视频并建立索引")
    from PIL import Image, ImageDraw, ImageFont
    from t0.media import FrameReader

    path = Path(params["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = int(params.get("columns", 4))
    thumb_w = int(params.get("thumb_width", 240))
    page_h = int(params.get("page_height", 2970))  # ~A4 @ 300dpi 缩小

    rows_data = session.scoped_shots(params)
    if not rows_data:
        raise ValueError("没有已完成镜头")

    reader = FrameReader(session.index)
    try:
        thumbs = []
        for shot in rows_data:
            mid = min((shot["start"] + shot["end"]) // 2, len(session.index.pts) - 1)
            frame = reader.read(mid)
            arr = frame.to_ndarray(format="rgb24")
            h, w = arr.shape[:2]
            tw = thumb_w
            th = int(h * tw / w)
            ys = (np.arange(th) * h // th).clip(0, h - 1)
            xs = (np.arange(tw) * w // tw).clip(0, w - 1)
            thumbs.append((arr[ys][:, xs], shot))
    finally:
        reader.close()

    import math
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 11)
        title_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    label_h = 22
    cell_h = max(t[0].shape[0] for t in thumbs) + label_h + 8
    per_page = max(1, (page_h - 80) // cell_h)
    pages = math.ceil(len(thumbs) / per_page)
    sheet_w = cols * (thumb_w + 8) + 16

    # 全部页合到一张长图，一次性存 PDF（Pillow 多页 PDF）
    all_pages = []
    for page in range(pages):
        img = Image.new("RGB", (sheet_w, min(page_h, (per_page) * cell_h + 80)), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((16, 16), f"CadensView 镜头联系表 — 第 {page+1}/{pages} 页", fill=(40,40,40), font=title_font)
        y_off = 60
        start = page * per_page
        for i, (thumb_arr, shot) in enumerate(thumbs[start:start+per_page]):
            row, col = divmod(i, cols)
            x = col * (thumb_w + 8) + 16
            y = y_off + row * cell_h
            img.paste(Image.fromarray(thumb_arr), (x, y))
            draw.rectangle([x, y, x+thumb_w-1, y+thumb_arr.shape[0]-1], outline=(200,200,200))
            label = f"#{shot['number']+1}  {shot['duration_s']:.1f}s"
            if shot.get("tags"):
                label += "  " + " ".join(shot["tags"])
            if shot.get("scale"):
                label += f"  [{shot['scale'].get('label_zh','')}]"
            draw.text((x, y + thumb_arr.shape[0] + 2), label, fill=(60,60,60), font=font)
        all_pages.append(img)

    if len(all_pages) == 1:
        all_pages[0].save(str(path), "PDF", resolution=150)
    else:
        all_pages[0].save(str(path), "PDF", save_all=True, append_images=all_pages[1:], resolution=150)
    return {"written": len(thumbs), "pages": pages, "path": str(path.resolve())}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        from tests.test_rpc import selftest
        selftest()
    else:
        serve()


