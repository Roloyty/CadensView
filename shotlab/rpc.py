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

from .domain import Project, Shot
from .history import History
from .storage import ProjectStore
from t0.media import FrameIndex, build_index

# 任务运行期间仍允许的请求（只读/控制类）。
BUSY_ALLOWED = {"ping", "project.state", "job.cancel"}


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

    def start_busy(self, name):
        if self.busy:
            raise ValueError(f"已有后台任务运行中（{self.busy}），请等待或取消")
        if name == "index.build":
            # 换片重建索引：先释放旧的项目文件句柄。
            self.close_current_store()
        self.busy = name

    def end_busy(self):
        self.busy = None

    def close_current_store(self):
        if self.store is not None:
            self.store.close()
            self.store = None

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
                shots.append({"id": shot.id, "number": number, "start": start, "end": end,
                              "note": shot.note, "pending": number >= completed,
                              "duration_s": round(duration_s, 4)})
        return {
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
        self.close_current_store()
        index = build_index(str(path), limit=limit, progress=progress, cancel=cancel)
        self.index = index
        self.video_path = str(path.resolve())
        self.project = None
        self.project_path = None
        self.stats = None
        self.dirty = False
        return {"index": self.index.__dict__, "video_path": self.video_path}

    def analysis_run(self, cancel=None, progress=None):
        """颜色时间轴 + 镜头统计；仅由后台任务线程调用（busy 保护下无并发写）。"""
        from shotlab.statistics import build_context
        from shotlab.statistics.colors import analyze_colors
        from shotlab.statistics.shots import shot_statistics
        if self.project is None or self.index is None:
            raise ValueError("先打开视频并建立索引")
        context = build_context(self.project, self.index)
        result = analyze_colors(context, cancel=cancel, progress=progress, include_summary=True)
        self.stats = shot_statistics(context)
        return dict(result, shot_statistics=self.stats)

    def new_project(self):
        if self.index is None:
            raise ValueError("先建立视频索引")
        self.index.verify_source()
        shot = Shot()
        self.project = Project(len(self.index.pts), [0, len(self.index.pts)], [shot],
                               selected_id=shot.id, pending_start=0)
        self.project.validate()
        self.history = History()
        self.close_current_store()
        self.project_path = None
        self.stats = None
        self.dirty = True
        return self.state()

    def open_project(self, path):
        store = ProjectStore(path)
        try:
            project, index, settings = store.load()
        except Exception:
            store.close()
            raise
        index.verify_source()
        self.close_current_store()
        self.store, self.project, self.index, self.settings = store, project, index, settings
        self.video_path = index.path
        self.project_path = str(store.path)
        self.history = History()
        self.stats = None
        self.dirty = False
        return self.state()

    def save_as(self, path):
        if self.project is None or self.index is None:
            raise ValueError("没有可保存的项目")
        target = Path(path).resolve()
        if self.project_path is not None and target == Path(self.project_path):
            return self.save()
        if self.store is not None:
            # 另存为：把当前内存态写进新副本，并把会话切换到新文件。
            if target.exists():
                target.unlink()
            self.store.backup(target, self.project, self.settings)
            self.close_current_store()
            self.store = ProjectStore(target)
        else:
            if target.exists():
                target.unlink()
            self.store = ProjectStore.create(target, self.index, self.project, self.settings)
        self.project_path = str(target)
        self.dirty = False
        return self.state()

    def save(self):
        if self.store is None:
            raise ValueError("项目尚未保存过文件")
        self.store.save(self.project, self.settings)
        self.dirty = False
        return self.state()

    def close_project(self):
        if self.store is not None:
            self.store.close()
        self.store = self.project = None
        self.project_path = None
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
    return session.save_as(params["path"])


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
    session.dirty = True
    return session.state()


@method("edit.redo")
def _edit_redo(session, params, emit=None):
    session.project = session.history.restore(redo=True)
    session.project.validate()
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
    context = build_context(session.project, session.index)
    return shot_statistics(context)


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
        progress=(lambda done, total: emit("analysis", done=done, total=total)) if emit else None)


@method("export.shots_csv")
def _export_shots_csv(session, params, emit=None):
    """镜头汇总 CSV（utf-8-sig，Excel 直接打开中文不乱码）。"""
    import csv
    if session.project is None or session.index is None:
        raise ValueError("先打开视频并建立索引")
    path = Path(params["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = session.state()["shots"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["镜头编号", "起始帧", "结束帧(不含)", "时长(秒)", "景别", "备注"])
        for shot in rows:
            writer.writerow([shot["number"] + 1, shot["start"], shot["end"],
                             f"{shot['duration_s']:.3f}",
                             session.labels.get(shot["id"], {}).get("scale", "") if hasattr(session, "labels") else "",
                             shot["note"]])
    return {"written": len(rows), "path": str(path.resolve())}


# -- protocol loop ----------------------------------------------------------

# 允许以后台任务方式运行的方法（长耗时、可取消）。
JOB_METHODS = {"index.build", "analysis.run"}


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
                    cancel=cancel,
                    progress=lambda done, total: emit("analysis", done=done, total=total))
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
            threading.Thread(target=run_job, args=(job_id, name, params.get("params") or {}),
                             name=f"job-{job_id}", daemon=True).start()
            write({"id": request.get("id"), "ok": True, "result": {"job_id": job_id, "method": name}})
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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        from tests.test_rpc import selftest
        selftest()
    else:
        serve()
