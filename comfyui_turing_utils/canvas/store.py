"""Instance-local immutable assets and published task versions (no tensor cache)."""

import json
import os
import re
import shutil
import threading
import uuid
from pathlib import Path

import folder_paths
import av
from PIL import Image


LOCK = threading.RLock()
KINDS = {"image", "video", "audio", "mask"}
EXTENSIONS = {"image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"},
              "video": {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"},
              "audio": {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}}


def probe_import(path, kind):
    if path.suffix.lower() not in EXTENSIONS.get(kind, set()):
        raise ValueError(f"Unsupported {kind} file extension")
    if kind == "image":
        with Image.open(path) as image:
            metadata = {"width": image.width, "height": image.height}
            image.verify()
            return metadata
    with av.open(str(path)) as container:
        streams = container.streams.video if kind == "video" else container.streams.audio
        if not streams:
            raise ValueError(f"File does not contain {kind}")
        stream = streams[0]
        metadata = {"audio": bool(container.streams.audio),
                    "duration": float(container.duration or 0) / av.time_base}
        if kind == "video":
            metadata.update(width=stream.width, height=stream.height, fps=float(stream.average_rate or 24))
        return metadata


def contained(root, relative):
    if not isinstance(relative, (str, Path)) or Path(relative).is_absolute() or "\\" in str(relative):
        raise ValueError("Canvas paths must be relative paths")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Canvas paths must be non-empty relative paths inside their owning directory")
    return path


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Project:
    def __init__(self, directory):
        self.directory = directory
        self.root = contained(folder_paths.get_output_directory(), directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets = self.root

    def state(self):
        path = contained(self.root, "project.json")
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"tasks": {}}

    def configure_cache(self, relative):
        cache_directory(relative)
        path = contained(self.root, "cache.json")
        value = {"directory": relative}
        if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != value:
            atomic_json(path, value)

    def cache(self):
        path = contained(self.root, "cache.json")
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"directory": self.directory}
        return cache_directory(value["directory"])

    def asset(self, asset_id):
        path = self.path(asset_id)
        data = json.loads(contained(self.root, asset_id + ".json").read_text(encoding="utf-8"))
        if data["id"] != asset_id:
            raise ValueError("Asset metadata does not match its filename")
        return data

    def path(self, asset_id):
        if not isinstance(asset_id, str) or not asset_id.startswith(("materials/", "generations/")):
            raise ValueError("Invalid project material filename")
        path = contained(self.root, asset_id)
        if not path.is_file():
            raise ValueError(f"Material file does not exist: {asset_id}")
        return path

    def reserve(self, extension, *, kind=None, name=None, prefix=None):
        if not re.fullmatch(r"\.[a-zA-Z0-9]{1,10}", extension):
            raise ValueError("Unsupported filename extension")
        if prefix is not None:
            if not isinstance(prefix, str) or not re.fullmatch(r"[^\\/:*?\"<>|.][^\\/:*?\"<>|]*", prefix) or prefix.endswith((".", " ")):
                raise ValueError("filename_prefix must be a non-empty filename, not a path")
            directory = contained(self.root, "generations/" + prefix)
            stem, numbered = prefix, True
        else:
            kind = kind or ("image" if extension.lower() == ".png" else "video")
            directory = contained(self.root, "materials/" + {"image": "images", "video": "videos", "audio": "audio"}[kind])
            stem = Path(name or "material").stem
            if not stem or re.search(r'[\\/:*?"<>|]', stem):
                raise ValueError("Invalid material filename")
            numbered = False
        directory.mkdir(parents=True, exist_ok=True)
        index = 1
        while True:
            filename = f"{stem}_{index:06d}{extension.lower()}" if numbered else f"{stem}{extension.lower()}"
            path = contained(directory, filename)
            try:
                with path.open("xb"):
                    pass
                return path.relative_to(self.root).as_posix(), path
            except FileExistsError:
                if numbered:
                    index += 1
                numbered = True

    def register(self, asset_id, path, kind, name, metadata=None):
        if kind not in KINDS or path.resolve() != contained(self.root, asset_id):
            raise ValueError("Invalid asset kind or location")
        data = {"id": asset_id, "file": asset_id, "kind": kind, "name": path.name,
                "metadata": metadata or {}}
        atomic_json(contained(self.root, asset_id + ".json"), data)
        return data

    def copy(self, source, kind):
        asset_id, target = self.reserve(source.suffix, kind=kind, name=source.name)
        temporary = target.with_suffix(target.suffix + ".partial")
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
            metadata = probe_import(target, kind)
            return self.register(asset_id, target, kind, source.name, metadata)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)

    def publish(self, task, asset, signature, snapshot):
        with LOCK:
            state = self.state()
            item = state["tasks"].setdefault(str(task), {"history": []})
            record = {"asset": asset, "signature": signature}
            if not any(r["asset"] == asset for r in item["history"]):
                item["history"].append(record)
            item.update(asset=asset, signature=signature)
            atomic_json(self.root / "project.json", state)
        return record

    def select(self, task, asset):
        with LOCK:
            state = self.state()
            item = state["tasks"][str(task)]
            record = next(r for r in item["history"] if r["asset"] == asset)
            item.update(asset=asset, signature=record["signature"])
            atomic_json(self.root / "project.json", state)

    def history(self, kind, prefix=None):
        if prefix is not None and (not prefix or re.search(r'[\\/:*?"<>|]', prefix) or prefix in (".", "..")):
            raise ValueError("Invalid filename prefix")
        directory = contained(self.root, "generations/" + prefix) if prefix else contained(
            self.root, "materials/" + {"image": "images", "video": "videos", "audio": "audio"}[kind])
        if not directory.exists():
            return []
        items = []
        for metadata in sorted(directory.glob("*.json")):
            if metadata.is_symlink():
                continue
            data = json.loads(metadata.read_text(encoding="utf-8"))
            if data.get("kind") == kind and contained(self.root, data["id"]).is_file():
                items.append({"id": data["id"], "name": data["name"]})
        return items


def cache_directory(relative):
    # Never use ComfyUI/temp: startup removes it.
    path = contained(Path(folder_paths.base_path) / ".cache", relative)
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_source(relative):
    # A browser cannot reveal a dragged file's server filesystem path.
    # Local-copy imports deliberately resolve only inside this instance's input.
    path = contained(folder_paths.get_input_directory(), relative)
    if not path.is_file():
        raise ValueError("Local-copy source must be a file in this instance's input directory")
    return path
