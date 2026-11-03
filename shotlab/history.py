"""Bounded session history. Snapshots include identity and observations together.

An edit may also carry an `extra` (before, after) pair for state kept beside the
Project, such as per-Shot annotations; restore exposes the matching side as
`last_extra` so the caller can put it back.
"""

from copy import deepcopy
from dataclasses import dataclass
from time import monotonic

from .domain import Project


@dataclass
class Edit:
    before: Project
    after: Project
    label: str
    key: str | None
    time: float
    extra: tuple | None = None


class History:
    def __init__(self, limit=100):
        self.limit = limit
        self.undo = []
        self.redo = []
        self.last_extra = None

    def record(self, before: Project, after: Project, label: str, key: str | None = None, extra=None):
        if before == after and (extra is None or extra[0] == extra[1]):
            return
        now = monotonic()
        if key and self.undo and self.undo[-1].key == key and now - self.undo[-1].time < 1:
            self.undo[-1].after = deepcopy(after)
            self.undo[-1].time = now
            if extra is not None:
                first = self.undo[-1].extra[0] if self.undo[-1].extra else extra[0]
                self.undo[-1].extra = (first, deepcopy(extra[1]))
        else:
            self.undo.append(Edit(before, deepcopy(after), label, key, now,
                                  None if extra is None else (extra[0], deepcopy(extra[1]))))
            del self.undo[:-self.limit]
        self.redo.clear()

    def restore(self, redo=False) -> Project:
        source, target = (self.redo, self.undo) if redo else (self.undo, self.redo)
        edit = source.pop()
        target.append(edit)
        self.last_extra = None if edit.extra is None else deepcopy(edit.extra[1 if redo else 0])
        return deepcopy(edit.after if redo else edit.before)

    def extend_analysis(self, cuts, complete=False, shot_ids=None):
        for edit in self.undo + self.redo:
            for project in (edit.before, edit.after):
                project.extend_analysis(cuts, complete, shot_ids)
