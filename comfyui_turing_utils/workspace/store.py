"""Project JSON owns selected content; SQLite indexes media and execution records."""

import json
import math
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
    @classmethod
    def create(cls, directory):
        root = inside(folder_paths.get_output_directory(), directory)
        with workflow_files.LOCK:
            if not root.is_dir() or any(root.iterdir()):
                raise ValueError("Select an existing empty folder to create a Canvas project")
            workflow_files.atomic_json(root / "canvas.json", {"format":3, "id":str(uuid.uuid4()), "revision":0, "cards":[], "selections":{}})
        return cls(directory)

    def __init__(self, directory):
        self.directory = directory
        self.root = inside(folder_paths.get_output_directory(), directory)
        with workflow_files.LOCK:
            path = inside(self.root, "canvas.json")
            if not path.is_file():
                raise ValueError("This folder has no canvas.json; create a project in an empty folder first")
            data = workflow_files.load_json(path)
            if not isinstance(data, dict) or data.get("format") != 3 or not isinstance(data.get("cards"), list) or not isinstance(data.get("selections"), dict) or not isinstance(data.get("id"), str) or type(data.get("revision")) is not int or data["revision"] < 0:
                raise ValueError("Invalid or unsupported Canvas project")
            uuid.UUID(data["id"])
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS materials (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
                    metadata TEXT NOT NULL, created INTEGER NOT NULL DEFAULT(unixepoch()));
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
        with workflow_files.LOCK:
            path = self.root / "canvas.json"
            canvas = workflow_files.load_json(path)
            return canvas["selections"]

    def patch(self, revision, changes):
        return workflow_files.patch(self.root, revision, changes)

    def select(self, node, content, expected=None):
        if set(content) - {"asset", "text"}:
            raise ValueError("Only material content is persistent; trim belongs to the current session")
        if "text" in content and (not isinstance(content["text"], str) or "asset" in content):
            raise ValueError("Text must be an inline string")
        with workflow_files.LOCK:
            selections = self.selections()
            revision = selections.get(node, {}).get("revision", 0)
            if expected is not None and revision != expected:
                raise Conflict("Material changed while editing; reload before replacing it")
            if content.get("asset"):
                self.asset(content["asset"])
            canvas = workflow_files.load_json(self.root / "canvas.json")
            canvas["selections"][node] = {**content, "revision": revision + 1}
            workflow_files.atomic_json(self.root / "canvas.json", canvas)
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
        metadata = {**metadata, "bytes": path.stat().st_size}
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

    def history(self, kind, offset=0, prefix="", limit=50):
        with self.connect() as db:
            # instr avoids treating user prefixes as SQL LIKE patterns.
            rows = db.execute(
                "SELECT id,name,kind FROM materials WHERE kind=? AND instr(name,?)=1 ORDER BY created DESC,id DESC LIMIT ? OFFSET ?",
                (kind, prefix, -1 if limit == 0 else max(1, min(int(limit), 1000)), max(0, int(offset)))).fetchall()
        return [dict(row) for row in rows]

    def settings(self):
        with workflow_files.LOCK:
            return {"name": self.root.name, "max_megapixels": 4.,
                    **workflow_files.load_json(self.root / "canvas.json").get("settings", {})}

    def save_settings(self, settings, revision):
        if set(settings) != {"name", "max_megapixels"} or not isinstance(settings["name"], str) or not settings["name"].strip():
            raise ValueError("Project needs a name and maximum megapixels")
        pixels = settings["max_megapixels"]
        if type(pixels) not in {int, float} or not math.isfinite(pixels) or pixels <= 0:
            raise ValueError("Maximum megapixels must be positive and finite")
        with workflow_files.LOCK:
            canvas = workflow_files.load_json(self.root / "canvas.json")
            if canvas["revision"] != revision:
                raise Conflict("Project changed; reopen before changing settings")
            canvas["settings"] = settings
            canvas["revision"] += 1
            workflow_files.atomic_json(self.root / "canvas.json", canvas)
        return canvas["revision"]

    def statistics(self):
        with self.connect() as db:
            media = db.execute("SELECT kind, count(*) AS count, sum(COALESCE(json_extract(metadata,'$.bytes'),0)) AS bytes FROM materials GROUP BY kind").fetchall()
        with workflow_files.LOCK:
            canvas = workflow_files.load_json(self.root / "canvas.json")
        counts = {kind: 0 for kind in ("image", "video", "audio", "text")}
        counts.update({row["kind"]: row["count"] for row in media})
        counts["text"] = sum("text" in value for value in canvas["selections"].values())
        return {"cards": len(canvas["cards"]), "materials": counts, "bytes": sum(row["bytes"] or 0 for row in media)}

    def begin_run(self, target, revision):
        run_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute("INSERT INTO runs(id,target,revision,status) VALUES(?,?,?,'prepared')", (run_id, target, revision))
        return run_id

    def finish_run(self, run_id, asset=None, text=None):
        with workflow_files.LOCK, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            run = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if run is None:
                raise ValueError("Unknown material execution")
            if run["status"] == "success":
                return bool(run["selected"])
            revision = self.selections().get(run["target"], {}).get("revision", 0)
            selected = revision == run["revision"]
            if selected:
                self.select(run["target"], {"text": text} if text is not None else {"asset": asset}, revision)
            db.execute("UPDATE runs SET status='success',asset=?,selected=? WHERE id=?", (asset, int(selected), run_id))
        return selected

    def run(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown execution")
        return dict(row)
