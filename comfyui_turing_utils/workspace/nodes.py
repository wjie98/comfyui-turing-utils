"""Material boundaries and private execution adapters using normal ComfyUI nodes."""

import copy
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import av
import nodes
import folder_paths
from comfy_api.latest import InputImpl, Types, io
from comfy_extras.nodes_video import CreateVideo
from comfy_extras.nodes_audio import load as load_audio

from .store import Project, inside
from .media import read_material
from .h3 import PrepareH3, FinishH3, SigmaRefiner
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

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {}
        optional = {"value": (cls.RETURN_TYPES[0], {"lazy": True}), "position": (POSITION, {"lazy": True})}
        inputs["stub_id"] = ("STRING", {"default": "", "hidden": True, "socketless": True})
        if cls.KIND == "text":
            inputs["text"] = ("STRING", {"default": "", "multiline": True})
        else:
            inputs["file"] = ("STRING", {"default": "", "tooltip": "Input-relative file; a connected computation takes precedence."})
        if cls.KIND == "video":
            del optional["value"]
            optional["images"] = ("IMAGE", {"lazy": True})
            optional["audio"] = ("AUDIO", {"lazy": True})
            inputs.update(fps=("FLOAT", {"default":30., "min":1., "max":120., "step":1.}),
                          bit_depth=(["auto", 8, 10],), color_space=(["sRGB", "HDR", "HDR PQ"],),
                          codec=(["none", *Types.VideoCodec.as_input()],))
        return {"required": inputs, "optional": optional}

    def check_lazy_status(self, **kwargs):
        return [name for name in ("value", "images", "audio") if name in kwargs and kwargs[name] is None]

    def read(self, file="", **kwargs):
        if self.KIND == "video" and kwargs.get("images") is not None:
            return (create_video(kwargs["images"], **kwargs),)
        if kwargs.get("value") is not None:
            return (kwargs["value"],)
        if self.KIND == "text":
            return (kwargs.get("text", ""),)
        relative, root = folder_paths.annotated_filepath(file)
        root = root or folder_paths.get_input_directory()
        if Path(root).resolve() not in {Path(folder_paths.get_input_directory()).resolve(), Path(folder_paths.get_output_directory()).resolve()}:
            raise ValueError("Material files must belong to this instance's input or output directory")
        path = inside(root, relative)
        if self.KIND == "video":
            return (InputImpl.VideoFromFile(str(path)),)
        if self.KIND == "image":
            return (nodes.LoadImage().load_image(file)[0],)
        waveform, rate = load_audio(str(path))
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": rate},)


def create_video(value, fps=30., audio=None, bit_depth="auto", color_space="sRGB", codec="none", **kwargs):
    return CreateVideo.execute(value, fps, audio, bit_depth, color_space, codec).result[0]


def read_selected(kind, directory, asset, start=0, end=0, include_audio=True):
    project = Project(directory)
    if project.asset(asset)["kind"] != kind:
        raise ValueError("Material kind does not match this node")
    if kind == "text":
        return (project.path(asset).read_text(encoding="utf-8"),)
    if end and end <= start:
        raise ValueError("End time must be after start time")
    if kind == "video":
        return (InputImpl.VideoFromFile(str(project.path(asset)), start_time=start, duration=end-start if end else 0),)
    images, audio, _ = read_material(project, {"asset": asset, "start": start,
        "duration": end - start if end else 0, "modality": "audio" if kind == "audio" else "av",
        "include_audio": include_audio, "max_megapixels": 4})
    return (images, audio) if kind == "video" else (audio,) if kind == "audio" else (images,)


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
                             "start": ("FLOAT",), "end": ("FLOAT",), "include_audio": ("BOOLEAN",)}}

    def read(self, directory="", asset="", run_id="", start=0, end=0, include_audio=True, text=""):
        if self.KIND == "text":
            return (text,)
        return read_selected(self.KIND, directory, asset, start, end, include_audio)


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
            temporary = path.with_name(path.stem + ".partial" + extension)
            metadata = {}
            try:
                if self.KIND == "image":
                    if len(value) != 1:
                        raise ValueError("Image material expects one image; use video material for a sequence")
                    Image.fromarray((value[0].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)).save(temporary)
                    metadata = {"width": value.shape[2], "height": value.shape[1]}
                elif self.KIND == "video":
                    video = create_video(value, fps, audio, bit_depth, color_space, codec)
                    video.save_to(str(temporary), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264, crf=19)
                    metadata = {"width": value.shape[2], "height": value.shape[1], "duration": len(value) / fps, "fps": fps, "audio": audio is not None}
                else:
                    waveform = value["waveform"][0].detach().cpu().float().numpy()
                    rate = value["sample_rate"]
                    layout = "mono" if waveform.shape[0] == 1 else "stereo"
                    if waveform.shape[0] not in {1, 2}:
                        raise ValueError("Audio material currently supports mono or stereo")
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
                temporary.replace(path)
                project.register(asset, self.KIND, metadata)
            finally:
                temporary.unlink(missing_ok=True)
        selected = project.finish_run(run_id, asset)
        return {"ui": {"material": [{"target": target, "asset": asset, "selected": selected, "run_id": run_id}]}, "result": (asset,)}


PUBLIC_NODES = {}
INTERNAL_NODES = {}
for _kind, _returns in MATERIAL_TYPES.items():
    _suffix = _kind.title()
    PUBLIC_NODES["TuringMaterial" + _suffix] = type("Material" + _suffix, (Material,), {"KIND": _kind, "RETURN_TYPES": _returns})
    INTERNAL_NODES["_TuringMaterialRead" + _suffix] = type("Read" + _suffix, (Read,), {"KIND": _kind, "RETURN_TYPES": _returns})
    INTERNAL_NODES["_TuringMaterialWrite" + _suffix] = type("Write" + _suffix, (Write,), {"KIND": _kind, "VALUE_TYPE": "IMAGE" if _kind == "video" else _returns[0]})
INTERNAL_NODES.update({"_TuringMaterialH3Prepare": PrepareH3,
                       "_TuringMaterialH3Finish": FinishH3,
                       "_TuringMaterialH3SigmaRefiner": SigmaRefiner})
PUBLIC_NODES.update(ENDPOINT_NODES)
