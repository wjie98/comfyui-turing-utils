"""Material boundaries and private execution adapters using normal ComfyUI nodes."""

from pathlib import Path
import folder_paths
from comfy_api.latest import Types
from .store import Project
from .endpoints import ENDPOINT_NODES, POSITION
from ..nodes import INTERNAL_NODE_NOTE
from .protocol import MATERIAL_TYPES
from .materials import (
    create_video,
    resolve_material,
    material_path,
    read_selected,
    save_media,
)


class Material:
    CATEGORY = "Turing Utils/Materials"
    FUNCTION = "read"
    KIND = "text"
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {}
        optional = {
            "value": (cls.RETURN_TYPES[0], {"lazy": True}),
            "position": (POSITION, {"lazy": True}),
        }
        inputs["stub_id"] = (
            "STRING",
            {"default": "", "hidden": True, "socketless": True},
        )
        if cls.KIND == "text":
            del optional["value"]
            inputs["text"] = ("STRING", {"default": "", "multiline": True})
        else:
            upload = (
                "video_upload"
                if cls.KIND == "video"
                else "audio_upload"
                if cls.KIND == "audio"
                else "image_upload"
            )
            inputs["audio" if cls.KIND == "audio" else "file"] = (
                "COMBO",
                {
                    upload: True,
                    "default": "",
                    "remote": {"route": "/turing/workspace/files/" + cls.KIND},
                },
            )
        if cls.KIND == "video":
            del optional["value"]
            optional["images"] = ("IMAGE", {"lazy": True})
            optional["audio"] = ("AUDIO", {"lazy": True})
            inputs.update(
                fps=("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                bit_depth=(["auto", 8, 10],),
                color_space=(["sRGB", "HDR", "HDR PQ"],),
                codec=(["none", *Types.VideoCodec.as_input()],),
            )
        return {"required": inputs, "optional": optional}

    def check_lazy_status(self, **kwargs):
        return [
            name
            for name in ("value", "images", "audio", "text")
            if name in kwargs and kwargs[name] is None
        ]

    @classmethod
    def IS_CHANGED(cls, file="", **kwargs):
        if cls.KIND == "audio":
            file = kwargs.get("audio", "")
        if file and kwargs.get("value") is None and kwargs.get("images") is None:
            stat = material_path(file).stat()
            return (stat.st_mtime_ns, stat.st_size)
        return None

    def read(self, file="", **kwargs):
        if self.KIND == "text":
            text = kwargs.get("text", "")
            return {"ui": {"material_text": [text]}, "result": (text,)}
        value, path = resolve_material(self.KIND, file, **kwargs)
        return self.saved_result(value, path)

    def saved_result(self, value, path):
        relative = path.relative_to(Path(folder_paths.get_output_directory()).resolve())
        data = {"material_file": [relative.as_posix() + " [output]"]}
        if self.KIND in {"image", "audio"}:
            # Native image/audio previews can reference the saved file without
            # encoding a second copy into ComfyUI's temporary directory.
            data["images" if self.KIND == "image" else "audio"] = [
                {
                    "filename": path.name,
                    "subfolder": relative.parent.as_posix(),
                    "type": "output",
                }
            ]
        return {"ui": data, "result": (value,)}


class Read:
    FUNCTION = "read"
    CATEGORY = ""
    DEV_ONLY = True
    DESCRIPTION = INTERNAL_NODE_NOTE

    @classmethod
    def INPUT_TYPES(cls):
        if cls.KIND == "text":
            return {"required": {"text": ("STRING",), "run_id": ("STRING",)}}
        return {
            "required": {
                "directory": ("STRING",),
                "asset": ("STRING",),
                "run_id": ("STRING",),
                "start": ("FLOAT",),
                "end": ("FLOAT",),
            }
        }

    def read(self, directory="", asset="", run_id="", start=0, end=0, text=""):
        if self.KIND == "text":
            return (text,)
        return read_selected(self.KIND, directory, asset, start, end)


class Write:
    FUNCTION = "write"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    CATEGORY = ""
    DEV_ONLY = True
    DESCRIPTION = INTERNAL_NODE_NOTE

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING",),
                "target": ("STRING",),
                "run_id": ("STRING",),
                "revision": ("INT",),
                "prefix": ("STRING",),
                "value": (cls.VALUE_TYPE,),
            },
            "optional": {
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 30}),
                "bit_depth": (["auto", 8, 10],),
                "color_space": (["sRGB", "HDR", "HDR PQ"],),
                "codec": (["none", *Types.VideoCodec.as_input()],),
            },
        }

    def write(
        self,
        directory,
        target,
        run_id,
        revision,
        prefix,
        value,
        audio=None,
        fps=30,
        bit_depth="auto",
        color_space="sRGB",
        codec="none",
    ):
        project = Project(directory)
        run = project.run(run_id)
        if run["target"] != target or run["revision"] != revision:
            raise ValueError("Execution destination mismatch")
        if run["status"] == "success":
            return {"ui": {"material": [run]}, "result": (run["asset"] or "",)}
        if self.KIND == "text":
            selected = project.finish_run(run_id, text=value)
            return {
                "ui": {
                    "material": [
                        {"target": target, "selected": selected, "run_id": run_id}
                    ]
                },
                "result": (value,),
            }
        else:
            extension = {"image": ".png", "video": ".mp4", "audio": ".wav"}[self.KIND]
            asset, path = project.reserve(self.KIND, extension, prefix)
            if self.KIND == "video":
                value = create_video(value, fps, audio, bit_depth, color_space, codec)
            metadata = save_media(self.KIND, value, path, codec)
            project.register(asset, self.KIND, metadata)
        selected = project.finish_run(run_id, asset)
        return {
            "ui": {
                "material": [
                    {
                        "target": target,
                        "asset": asset,
                        "selected": selected,
                        "run_id": run_id,
                    }
                ]
            },
            "result": (asset,),
        }


PUBLIC_NODES = {}


INTERNAL_NODES = {}


for _kind, _returns in MATERIAL_TYPES.items():
    _suffix = _kind.title()
    PUBLIC_NODES["TuringMaterial" + _suffix] = type(
        "Material" + _suffix, (Material,), {"KIND": _kind, "RETURN_TYPES": _returns}
    )
    INTERNAL_NODES["_TuringMaterialRead" + _suffix] = type(
        "Read" + _suffix, (Read,), {"KIND": _kind, "RETURN_TYPES": _returns}
    )
    INTERNAL_NODES["_TuringMaterialWrite" + _suffix] = type(
        "Write" + _suffix,
        (Write,),
        {"KIND": _kind, "VALUE_TYPE": "IMAGE" if _kind == "video" else _returns[0]},
    )


PUBLIC_NODES.update(ENDPOINT_NODES)
