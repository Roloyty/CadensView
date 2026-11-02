"""JSON-RPC sidecar for native GUI clients; one request per line on stdin.

The protocol is synchronous: the GUI sends {"id", "method", "params"} and reads
matching {"id", "ok", "result"|"error"} responses. Long jobs emit progress
notifications ({"event": "progress", ...}) before their final response. All
state (index, project, history, store) lives here so the domain rules, undo and
storage stay inside the tested core; the GUI is a pure view.
"""

import json
import sys
from pathlib import Path

from .domain import Project, Shot
from .history import History
from .storage import ProjectStore
from t0.media import FrameIndex, build_index


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

    def build_index(self, video_path, limit=None, emit=None):
        path = Path(video_path)
        if not path.is_file():
            raise ValueError(f"视频文件不存在: {video_path}")
        progress = (lambda count, seconds: emit("index", done=count, total=seconds)) if emit else None
        self.index = build_index(str(path), limit=limit, progress=progress)
        self.video_path = str(path.resolve())
        self.project = None
        self.store = None
        self.project_path = None
        return {"index": self.index.__dict__, "video_path": self.video_path}

    def new_project(self):
        if self.index is None:
            raise ValueError("先建立视频索引")
        self.index.verify_source()
        shot = Shot()
        self.project = Project(len(self.index.pts), [0, len(self.index.pts)], [shot],
                               selected_id=shot.id, pending_start=0)
        self.project.validate()
        self.history = History()
        self.store = None
        self.project_path = None
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
        self.store, self.project, self.index, self.settings = store, project, index, settings
        self.video_path = index.path
        self.project_path = str(store.path)
        self.history = History()
        self.dirty = False
        return self.state()

    def save_as(self, path):
        if self.project is None or self.index is None:
            raise ValueError("没有可保存的项目")
        if self.store is not None:
            raise ValueError("项目已存在文件，请直接保存或另存备份")
        self.store = ProjectStore.create(path, self.index, self.project, self.settings)
        self.project_path = str(self.store.path)
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
        self.dirty = True
        return self.state()

    def select(self, frame=None, shot_id=None, record=False):
        if shot_id:
            self.project.row(shot_id)  # ValueError when unknown
            if frame is None:
                frame = self.project.interval(self.project.row(shot_id))[0]
        if frame is not None:
            self.project.shot_at(frame)  # range check
            self.project.current_frame = int(frame)
        if shot_id:
            self.project.selected_id = shot_id
        self.dirty = True
        return self.state()

    def apply_detection(self, result_dir):
        result = json.loads((Path(result_dir) / "result.json").read_text(encoding="utf-8"))
        cuts = sorted({int(b["frame"]) for b in result["boundaries"]})
        if self.project.pending_start is None:
            raise ValueError("分析已完成，不能覆盖已有镜头")
        ids = self.project.extend_analysis(cuts, complete=True)
        self.history.extend_analysis(cuts, complete=True, shot_ids=ids)
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
def _ping(session, params):
    return {"version": "0.1.0"}


@method("index.build")
def _index_build(session, params):
    return session.build_index(params["video_path"], limit=params.get("limit"))


@method("index.verify")
def _index_verify(session, params):
    session.index.verify_source()
    return {"ok": True}


@method("index.get")
def _index_get(session, params):
    if session.index is None:
        raise ValueError("尚未建立索引")
    return {"index": session.index.__dict__}


@method("project.new")
def _project_new(session, params):
    return session.new_project()


@method("project.open")
def _project_open(session, params):
    return session.open_project(params["path"])


@method("project.save")
def _project_save(session, params):
    return session.save()


@method("project.save_as")
def _project_save_as(session, params):
    return session.save_as(params["path"])


@method("project.close")
def _project_close(session, params):
    return session.close_project()


@method("project.state")
def _project_state(session, params):
    return session.state()


@method("edit.split")
def _edit_split(session, params):
    return session._edit("拆分镜头", mutate=lambda: session.project.split(int(params["frame"])))


@method("edit.merge")
def _edit_merge(session, params):
    return session._edit("合并镜头", mutate=lambda: session.project.merge(int(params["row"])))


@method("edit.move_boundary")
def _edit_move(session, params):
    return session._edit("移动边界",
                         mutate=lambda: session.project.move_boundary(int(params["boundary"]), int(params["frame"])))


@method("edit.set_note")
def _edit_note(session, params):
    shot_id, text = params["shot_id"], params.get("text", "")
    return session._edit("镜头笔记", key=f"note:{shot_id}",
                         mutate=lambda: session.project.set_note(shot_id, text))


@method("edit.select")
def _edit_select(session, params):
    return session.select(frame=params.get("frame"), shot_id=params.get("shot_id"))


@method("edit.undo")
def _edit_undo(session, params):
    session.project = session.history.restore(redo=False)
    session.project.validate()
    session.dirty = True
    return session.state()


@method("edit.redo")
def _edit_redo(session, params):
    session.project = session.history.restore(redo=True)
    session.project.validate()
    session.dirty = True
    return session.state()


@method("project.apply_detection")
def _apply_detection(session, params):
    return session.apply_detection(params["result_dir"])


# -- protocol loop ----------------------------------------------------------

def serve(stdin=sys.stdin, stdout=sys.stdout, session=None):
    if hasattr(stdin, "reconfigure"):
        stdin.reconfigure(encoding="utf-8")
    if hasattr(stdout, "reconfigure"):
        stdout.reconfigure(encoding="utf-8")
    session = session if session is not None else Session()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            handler = METHODS[request.get("method")]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            _write(stdout, {"id": request.get("id") if isinstance(request, dict) else None,
                            "ok": False, "error": {"message": f"bad request: {error}"}})
            continue

        def emit(job, **values):
            _write(stdout, {"event": "progress", "job": job, **values})

        try:
            result = handler(session, request.get("params") or {})
            _write(stdout, {"id": request.get("id"), "ok": True, "result": result})
        except Exception as error:  # domain errors are part of the contract
            _write(stdout, {"id": request.get("id"), "ok": False,
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
