"""Material boundaries and private execution adapters using normal ComfyUI nodes."""

import copy
import hashlib
import math
from pathlib import Path
import shutil
import uuid

import numpy as np
import av
import nodes
import folder_paths
from comfy_api.latest import InputImpl, Types, io
from comfy_extras.nodes_video import CreateVideo
from comfy_extras.nodes_audio import load as load_audio

from .store import Project, inside
from .media import read_image, read_audio
from .endpoints import ENDPOINT_NODES, POSITION
from ..nodes import INTERNAL_NODE_NOTE
from .protocol import MATERIAL_TYPES


def fresh_type(name):
    """Isolated derived classes: never mutate third-party classes or the executor."""
    original = nodes.NODE_CLASS_MAPPINGS[name]
    key = "_TuringMaterialFresh_" + name
    if key not in nodes.NODE_CLASS_MAPPINGS:
        fingerprint = classmethod(lambda cls, **kwargs: float("nan"))
        attributes = {"IS_CHANGED": fingerprint, "DEV_ONLY": True, "CATEGORY": "", "DESCRIPTION": INTERNAL_NODE_NOTE}
        if issubclass(original, io.ComfyNode):
            attributes["fingerprint_inputs"] = fingerprint
            def schema(cls):
                result = copy.deepcopy(original.define_schema())
                result.node_id = key
                result.display_name = f"{result.display_name or name} Fresh (Internal)"
                result.description = INTERNAL_NODE_NOTE + (result.description or "")
                result.category = ""
                result.is_dev_only = True
                return result
            attributes["define_schema"] = classmethod(schema)
        nodes.NODE_CLASS_MAPPINGS[key] = type(key, (original,), attributes)
        nodes.NODE_DISPLAY_NAME_MAPPINGS[key] = f"{name} Fresh (Internal)"
    return key


class Material:
    CATEGORY = "Turing Utils/Materials"
    FUNCTION = "read"
    KIND = "text"
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {}
        optional = {"value": (cls.RETURN_TYPES[0], {"lazy": True}), "position": (POSITION, {"lazy": True})}
        inputs["stub_id"] = ("STRING", {"default": "", "hidden": True, "socketless": True})
        if cls.KIND == "text":
            del optional["value"]
            inputs["text"] = ("STRING", {"default": "", "multiline": True})
        else:
            upload = "video_upload" if cls.KIND == "video" else "audio_upload" if cls.KIND == "audio" else "image_upload"
            inputs["audio" if cls.KIND == "audio" else "file"] = ("COMBO", {upload: True, "default": "", "remote": {
                "route": "/turing/workspace/files/" + cls.KIND}})
        if cls.KIND == "video":
            del optional["value"]
            optional["images"] = ("IMAGE", {"lazy": True})
            optional["audio"] = ("AUDIO", {"lazy": True})
            inputs.update(fps=("FLOAT", {"default":30., "min":1., "max":120., "step":1.}),
                          bit_depth=(["auto", 8, 10],), color_space=(["sRGB", "HDR", "HDR PQ"],),
                          codec=(["none", *Types.VideoCodec.as_input()],))
        return {"required": inputs, "optional": optional}

    def check_lazy_status(self, **kwargs):
        return [name for name in ("value", "images", "audio", "text") if name in kwargs and kwargs[name] is None]

    @classmethod
    def IS_CHANGED(cls, file="", **kwargs):
        if cls.KIND == "audio":
            file = kwargs.get("audio", "")
        if file and kwargs.get("value") is None and kwargs.get("images") is None:
            stat = material_path(file).stat()
            return (stat.st_mtime_ns, stat.st_size)
        return None

    def read(self, file="", **kwargs):
        if self.KIND == "audio":
            file = kwargs.get("audio", file)
        if self.KIND == "text":
            text = kwargs.get("text", "")
            return {"ui": {"material_text": [text]}, "result": (text,)}
        if self.KIND == "video" and kwargs.get("images") is not None:
            value = create_video(kwargs["images"], **kwargs)
        elif kwargs.get("value") is not None:
            value = kwargs["value"]
        else:
            if not file:
                raise ValueError(f"Choose a {self.KIND} file or connect material content")
            path = material_path(file)
            if self.KIND == "video":
                value = InputImpl.VideoFromFile(str(path))
            elif self.KIND == "image":
                value = nodes.LoadImage().load_image(file)[0]
            else:
                waveform, rate = load_audio(str(path))
                value = {"waveform": waveform.unsqueeze(0), "sample_rate": rate}
            output_root = Path(folder_paths.get_output_directory()).resolve()
            if not path.is_relative_to(output_root):
                stat = path.stat()
                digest = hashlib.sha256(f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[:12]
                destination = inside(output_root, f"materials/{self.KIND}/{path.stem}_{digest}{path.suffix.lower()}")
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(uuid.uuid4().hex + ".partial")
                    try:
                        shutil.copyfile(path, temporary)
                        temporary.replace(destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                path = destination
            return self.saved_result(value, path)
        extension = {"image": ".png", "video": ".mp4", "audio": ".wav"}[self.KIND]
        relative = f"materials/{self.KIND}/{self.KIND}_{uuid.uuid4().hex[:12]}{extension}"
        path = inside(folder_paths.get_output_directory(), relative)
        save_media(self.KIND, value, path, codec=kwargs.get("codec", "none"))
        return self.saved_result(value, path)

    def saved_result(self, value, path):
        relative = path.relative_to(Path(folder_paths.get_output_directory()).resolve())
        data = {"material_file": [relative.as_posix() + " [output]"]}
        if self.KIND in {"image", "audio"}:
            # Native image/audio previews can reference the saved file without
            # encoding a second copy into ComfyUI's temporary directory.
            data["images" if self.KIND == "image" else "audio"] = [{"filename":path.name,
                "subfolder":relative.parent.as_posix(), "type":"output"}]
        return {"ui": data, "result": (value,)}


def material_path(file):
    relative, root = folder_paths.annotated_filepath(file)
    root = root or folder_paths.get_input_directory()
    if Path(root).resolve() not in {Path(folder_paths.get_input_directory()).resolve(), Path(folder_paths.get_output_directory()).resolve()}:
        raise ValueError("Material files must belong to this instance's input or output directory")
    return inside(root, relative)


def save_media(kind, value, path, codec="none"):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    try:
        if kind == "image":
            if len(value) != 1:
                raise ValueError("Image material expects one image; use video material for a sequence")
            saver = nodes.SaveImage()
            saver.output_dir = str(path.parent)
            result = saver.save_images(value, temporary.stem)
            (path.parent / result["ui"]["images"][0]["filename"]).replace(temporary)
            metadata = {"width": value.shape[2], "height": value.shape[1]}
        elif kind == "video":
            value.save_to(str(temporary), format=Types.VideoContainer.MP4,
                          codec=Types.VideoCodec.AUTO if codec == "none" else Types.VideoCodec(codec), crf=19)
            width, height = value.get_dimensions()
            metadata = {"width": width, "height": height, "duration": value.get_duration(),
                        "fps": float(value.get_frame_rate()), "bit_depth": value.get_bit_depth(),
                        "color_space": value.get_color_space()}
        else:
            waveform = value["waveform"][0].detach().cpu().float().numpy()
            rate = value["sample_rate"]
            if waveform.shape[0] not in {1, 2}:
                raise ValueError("Audio material currently supports mono or stereo")
            layout = "mono" if waveform.shape[0] == 1 else "stereo"
            with av.open(str(temporary), "w", format="wav") as output:
                stream = output.add_stream("pcm_s16le", rate=rate)
                stream.layout = layout
                frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(waveform), format="fltp", layout=layout)
                frame.sample_rate = rate
                for packet in stream.encode(frame):
                    output.mux(packet)
                for packet in stream.encode(None):
                    output.mux(packet)
            metadata = {"duration": waveform.shape[1] / rate}
        metadata["bytes"] = temporary.stat().st_size
        temporary.replace(path)
        return metadata
    finally:
        temporary.unlink(missing_ok=True)


def create_video(value, fps=30., audio=None, bit_depth="auto", color_space="sRGB", codec="none", **kwargs):
    return CreateVideo.execute(value, fps, audio, bit_depth, color_space, codec).result[0]


def read_selected(kind, directory, asset, start=0, end=0):
    project = Project(directory)
    if project.asset(asset)["kind"] != kind:
        raise ValueError("Material kind does not match this node")
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < 0 or (end and end <= start):
        raise ValueError("End time must be after start time")
    maximum = project.settings()["max_megapixels"]
    if kind == "video":
        video = InputImpl.VideoFromFile(str(project.path(asset)), start_time=start, duration=end-start if end else 0)
        width, height = video.get_dimensions()
        if width * height > maximum * 1024 * 1024:
            components = video.get_components()
            scale = math.sqrt(maximum * 1024 * 1024 / (width * height))
            # Floor dimensions so native rounding cannot exceed the pixel cap.
            height, width = components.images.shape[1:3]
            width, height = max(1, int(width * scale)), max(1, int(height * scale))
            components.images = nodes.ImageScale().upscale(components.images, "area", width, height, "disabled")[0]
            if components.alpha is not None:
                components.alpha = nodes.ImageScale().upscale(components.alpha.unsqueeze(-1), "area", width, height, "disabled")[0].squeeze(-1)
            video = InputImpl.VideoFromComponents(components, bit_depth=video.get_bit_depth(), color_space=video.get_color_space())
        return (video,)
    path = project.path(asset)
    return (read_audio(path, start, end),) if kind == "audio" else (read_image(path, maximum),)


class Read:
    FUNCTION = "read"
    CATEGORY = ""
    DEV_ONLY = True
    DESCRIPTION = INTERNAL_NODE_NOTE

    @classmethod
    def INPUT_TYPES(cls):
        if cls.KIND == "text":
            return {"required": {"text": ("STRING",), "run_id": ("STRING",)}}
        return {"required": {"directory": ("STRING",), "asset": ("STRING",), "run_id": ("STRING",),
                             "start": ("FLOAT",), "end": ("FLOAT",)}}

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
        return {"required": {"directory": ("STRING",), "target": ("STRING",), "run_id": ("STRING",),
            "revision": ("INT",), "prefix": ("STRING",), "value": (cls.VALUE_TYPE,)},
            "optional": {"audio": ("AUDIO",), "fps": ("FLOAT", {"default": 30}),
                         "bit_depth": (["auto", 8, 10],), "color_space": (["sRGB", "HDR", "HDR PQ"],),
                         "codec": (["none", *Types.VideoCodec.as_input()],)}}

    def write(self, directory, target, run_id, revision, prefix, value, audio=None, fps=30, bit_depth="auto", color_space="sRGB", codec="none"):
        project = Project(directory)
        run = project.run(run_id)
        if run["target"] != target or run["revision"] != revision:
            raise ValueError("Execution destination mismatch")
        if run["status"] == "success":
            return {"ui": {"material": [run]}, "result": (run["asset"] or "",)}
        if self.KIND == "text":
            selected = project.finish_run(run_id, text=value)
            return {"ui": {"material": [{"target": target, "selected": selected, "run_id": run_id}]}, "result": (value,)}
        else:
            extension = {"image": ".png", "video": ".mp4", "audio": ".wav"}[self.KIND]
            asset, path = project.reserve(self.KIND, extension, prefix)
            if self.KIND == "video":
                value = create_video(value, fps, audio, bit_depth, color_space, codec)
            metadata = save_media(self.KIND, value, path, codec)
            project.register(asset, self.KIND, metadata)
        selected = project.finish_run(run_id, asset)
        return {"ui": {"material": [{"target": target, "asset": asset, "selected": selected, "run_id": run_id}]}, "result": (asset,)}


PUBLIC_NODES = {}
INTERNAL_NODES = {}
for _kind, _returns in MATERIAL_TYPES.items():
    _suffix = _kind.title()
    PUBLIC_NODES["TuringMaterial" + _suffix] = type("Material" + _suffix, (Material,), {"KIND": _kind, "RETURN_TYPES": _returns})
    INTERNAL_NODES["_TuringMaterialRead" + _suffix] = type("Read" + _suffix, (Read,), {"KIND": _kind, "RETURN_TYPES": _returns})
    INTERNAL_NODES["_TuringMaterialWrite" + _suffix] = type("Write" + _suffix, (Write,), {"KIND": _kind, "VALUE_TYPE": "IMAGE" if _kind == "video" else _returns[0]})
PUBLIC_NODES.update(ENDPOINT_NODES)
