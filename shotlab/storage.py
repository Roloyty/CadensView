"""One SQLite project, atomic saves and explicit concurrent-writer detection."""

import json
from pathlib import Path
import sqlite3

from t0.media import FrameIndex
from .domain import Project, Shot


class SaveConflict(RuntimeError):
    pass


class ProjectStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.connection = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=2)
        try:
            if self.connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise ValueError("Unsupported project version; file left unchanged")
            self.revision = self.connection.execute("SELECT revision FROM project").fetchone()[0]
        except BaseException:
            # A file that is not a usable project must not stay open: on Windows the handle
            # keeps it locked until garbage collection.
            self.connection.close()
            raise

    @classmethod
    def create(cls, path: str | Path, index: FrameIndex, project: Project | None = None, settings: dict | None = None):
        if not index.complete:
            raise ValueError("A project requires the complete source index, not a test prefix")
        index.verify_source()
        project = project if project is not None else Project.new(len(index.pts))
        project.validate()
        if project.frame_count != len(index.pts):
            raise ValueError("Project partition does not match the source index")
        settings = {"audio_track": 0 if index.metadata["audio_tracks"] else -1,
                    "subtitle_path": "", **(settings or {})}
        settings = dict(settings, pending_start=project.pending_start)
        path = Path(path).resolve()
        # Exclusive creation also prevents accidental replacement of a video or
        # an existing project, regardless of extension.
        with path.open("xb"):
            pass
        connection = sqlite3.connect(path)
        created = False
        try:
            with connection:
                connection.executescript("""
                    PRAGMA user_version=1;
                    CREATE TABLE project (
                        id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                        media_index TEXT NOT NULL, current_frame INTEGER NOT NULL,
                        selected_id TEXT NOT NULL, settings TEXT NOT NULL
                    );
                    CREATE TABLE shots (
                        id TEXT PRIMARY KEY, position INTEGER UNIQUE NOT NULL,
                        start_frame INTEGER UNIQUE NOT NULL, note TEXT NOT NULL
                    );
                """)
                connection.execute("INSERT INTO project VALUES (?,0,?,?,?,?)", (
                    project.id, json.dumps(index.__dict__, ensure_ascii=False), project.current_frame,
                    project.selected_id, json.dumps(settings, ensure_ascii=False)))
                connection.executemany("INSERT INTO shots VALUES (?,?,?,?)", [
                    (shot.id, number, project.boundaries[number], shot.note)
                    for number, shot in enumerate(project.shots)])
            created = True
        finally:
            connection.close()
            if not created:
                path.unlink()
        return cls(path)

    def load(self) -> tuple[Project, FrameIndex, dict]:
        row = self.connection.execute("SELECT id,revision,media_index,current_frame,selected_id,settings FROM project").fetchone()
        project_id, self.revision, raw_index, current, selected, settings = row
        index = FrameIndex(**json.loads(raw_index))
        rows = self.connection.execute("SELECT id,start_frame,note FROM shots ORDER BY position").fetchall()
        settings = json.loads(settings)
        project = Project(len(index.pts), [r[1] for r in rows] + [len(index.pts)],
                          [Shot(r[0], r[2]) for r in rows], project_id, current, selected,
                          settings.get('pending_start'))
        project.validate()
        return project, index, settings

    def save(self, project: Project, settings: dict, index: FrameIndex | None = None):
        project.validate()
        settings = dict(settings, pending_start=project.pending_start)
        with self.connection:
            changed = self.connection.execute(
                "UPDATE project SET revision=revision+1,current_frame=?,selected_id=?,settings=? WHERE revision=?",
                (project.current_frame, project.selected_id, json.dumps(settings, ensure_ascii=False), self.revision))
            if changed.rowcount != 1:
                raise SaveConflict("Project was saved by another instance; reopen it or save a backup")
            if index is not None:
                self.connection.execute("UPDATE project SET media_index=?", (json.dumps(index.__dict__, ensure_ascii=False),))
            self.connection.execute("DELETE FROM shots")
            self.connection.executemany("INSERT INTO shots VALUES (?,?,?,?)", [
                (shot.id, number, project.boundaries[number], shot.note)
                for number, shot in enumerate(project.shots)])
        self.revision += 1

    def backup(self, path: str | Path, project: Project, settings: dict):
        path = Path(path).resolve()
        with path.open("xb"):
            pass
        target = sqlite3.connect(path)
        try:
            self.connection.backup(target)
        finally:
            target.close()
        # Save the current in-memory state into the copy, independently of
        # whether the original file is writable or has a revision conflict.
        snapshot = ProjectStore(path)
        try:
            snapshot.save(project, settings)
        finally:
            snapshot.close()

    def close(self):
        self.connection.close()
