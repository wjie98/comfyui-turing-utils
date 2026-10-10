"""Instance-owned directory browsing and empty-tree removal."""

from pathlib import Path

import folder_paths

from .store import inside


def list_directory(relative=""):
    root = Path(folder_paths.get_output_directory()).resolve()
    parent = inside(root, relative) if relative else root
    items = []
    for path in sorted(
        parent.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold())
    ):
        if path.is_symlink() or path.name.startswith("."):
            continue
        items.append(
            {
                "name": path.name,
                "path": path.relative_to(root).as_posix(),
                "directory": path.is_dir(),
                "project": path.is_dir() and (path / "canvas.json").is_file(),
                "bytes": path.stat().st_size if path.is_file() else 0,
            }
        )
    return {
        "items": items,
        "path": relative,
        "empty": not any(parent.iterdir()),
        "project": (parent / "canvas.json").is_file(),
    }


def remove_empty_directory(relative):
    root = Path(folder_paths.get_output_directory()).resolve()
    target = inside(root, relative)
    raw = root / relative
    if any(
        path.is_symlink() for path in [raw, *raw.parents] if path.is_relative_to(root)
    ):
        raise ValueError("Cannot delete through a symbolic link")
    directories = []

    def inspect(path):
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Only an empty directory tree may be deleted")
        for child in path.iterdir():
            inspect(child)
        directories.append(path)

    inspect(target)
    for path in directories:
        path.rmdir()
