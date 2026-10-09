"""Files own material content; SQLite owns revisions and concurrent selection."""

import json
import math
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

import folder_paths
from . import workflow_files


def inside(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or Path(relative).is_absolute():
        raise ValueError("Use a nonempty project-relative path")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Path escapes its owning directory")
    return path


class Conflict(ValueError):
    pass


class Project:
    def __init__(self, directory):
        self.directory = directory
        self.root = inside(folder_paths.get_output_directory(), directory)
        self.root.mkdir(parents=True, exist_ok=True)
        with workflow_files.LOCK:
            path = inside(self.root, "canvas.json")
            if not path.exists():
                workflow_files.atomic_json(path, {"format":2, "revision":0, "cards":[]})
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS materials (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
                    metadata TEXT NOT NULL, created INTEGER NOT NULL DEFAULT(unixepoch()));
                CREATE TABLE IF NOT EXISTS selections (node TEXT PRIMARY KEY, revision INTEGER NOT NULL, content TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, target TEXT NOT NULL,
                    revision INTEGER NOT NULL, status TEXT NOT NULL, asset TEXT, selected INTEGER DEFAULT 0);
                CREATE INDEX IF NOT EXISTS material_kind ON materials(kind, created DESC);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(inside(self.root, "workspace.sqlite3"), timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def document(self):
        with workflow_files.LOCK:
            return {**workflow_files.document(self.root), "selections": self.selections()}

    def selections(self):
        with self.connect() as db:
            return {r["node"]: {**json.loads(r["content"]), "revision": r["revision"]}
                    for r in db.execute("SELECT * FROM selections")}

    def patch(self, revision, changes):
        return workflow_files.patch(self.root, revision, changes)

    def select(self, node, content, expected=None):
        start, end = float(content.get("start", 0)), float(content.get("end", 0))
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < 0 or (end and end <= start):
            raise ValueError("Use a nonnegative start and an end after start (0 means file end)")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT revision FROM selections WHERE node=?", (node,)).fetchone()
            revision = row[0] if row else 0
            if expected is not None and revision != expected:
                raise Conflict("Material changed while editing; reload before replacing it")
            if content.get("asset") and not db.execute("SELECT 1 FROM materials WHERE id=?", (content["asset"],)).fetchone():
                raise ValueError("Unknown material")
            db.execute("INSERT OR REPLACE INTO selections VALUES(?,?,?)", (node, revision + 1, json.dumps(content)))
        return revision + 1

    def reserve(self, kind, suffix, prefix):
        if kind not in {"text", "image", "video", "audio"} or not re.fullmatch(r"\.[a-zA-Z0-9]+", suffix):
            raise ValueError("Invalid material format")
        if not prefix or re.search(r'[\\/:*?"<>|\x00-\x1f]', prefix) or prefix in {".", ".."}:
            raise ValueError("Prefix must be a filename, not a path")
        relative = f"materials/{kind}/{prefix}_{uuid.uuid4().hex[:12]}{suffix.lower()}"
        path = inside(self.root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        return relative, path

    def register(self, asset, kind, metadata):
        path = inside(self.root, asset)
        if not path.is_file():
            raise ValueError("Material file is missing")
        with self.connect() as db:
            db.execute("INSERT INTO materials(id,kind,name,metadata) VALUES(?,?,?,?)",
                       (asset, kind, path.name, json.dumps(metadata)))

    def asset(self, asset):
        with self.connect() as db:
            row = db.execute("SELECT * FROM materials WHERE id=?", (asset,)).fetchone()
        if row is None:
            raise ValueError("Unknown material")
        return {**dict(row), "metadata": json.loads(row["metadata"])}

    def path(self, asset):
        self.asset(asset)
        path = inside(self.root, asset)
        if not path.is_file():
            raise ValueError("Material file is missing")
        return path

    def history(self, kind, offset=0, prefix=""):
        with self.connect() as db:
            # instr avoids treating user prefixes as SQL LIKE patterns.
            rows = db.execute("SELECT id,name,kind FROM materials WHERE kind=? AND instr(name,?)=1 ORDER BY created DESC,id DESC LIMIT 50 OFFSET ?",
                              (kind, prefix, max(0, int(offset)))).fetchall()
        return [dict(row) for row in rows]

    def begin_run(self, target, revision):
        run_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute("INSERT INTO runs(id,target,revision,status) VALUES(?,?,?,'prepared')", (run_id, target, revision))
        return run_id

    def finish_run(self, run_id, asset):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            run = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if run is None:
                raise ValueError("Unknown material execution")
            if run["status"] == "success":
                return bool(run["selected"])
            row = db.execute("SELECT revision FROM selections WHERE node=?", (run["target"],)).fetchone()
            revision = row[0] if row else 0
            selected = revision == run["revision"]
            if selected:
                db.execute("INSERT OR REPLACE INTO selections VALUES(?,?,?)",
                           (run["target"], revision + 1, json.dumps({"asset": asset})))
            db.execute("UPDATE runs SET status='success',asset=?,selected=? WHERE id=?", (asset, int(selected), run_id))
        return selected

    def run(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown execution")
        return dict(row)

    def text(self, text, prefix="text"):
        asset, path = self.reserve("text", ".txt", prefix)
        temporary = path.with_suffix(".partial")
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self.register(asset, "text", {})
        return asset
