"""Material resolution, serialization and selected-content decoding."""

import math
import hashlib
import shutil
import uuid
from pathlib import Path
import numpy as np
import av
import nodes
import folder_paths
from comfy_api.latest import InputImpl, Types
from comfy_extras.nodes_video import CreateVideo
from comfy_extras.nodes_audio import load as load_audio
from .store import Project, inside
from ..media.files import read_image, read_audio


def material_path(file):
    relative, root = folder_paths.annotated_filepath(file)
    root = root or folder_paths.get_input_directory()
    if Path(root).resolve() not in {
        Path(folder_paths.get_input_directory()).resolve(),
        Path(folder_paths.get_output_directory()).resolve(),
    }:
        raise ValueError(
            "Material files must belong to this instance's input or output directory"
        )
    return inside(root, relative)


def save_media(kind, value, path, codec="none"):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    try:
        if kind == "image":
            if len(value) != 1:
                raise ValueError(
                    "Image material expects one image; use video material for a sequence"
                )
            saver = nodes.SaveImage()
            saver.output_dir = str(path.parent)
            result = saver.save_images(value, temporary.stem)
            (path.parent / result["ui"]["images"][0]["filename"]).replace(temporary)
            metadata = {"width": value.shape[2], "height": value.shape[1]}
        elif kind == "video":
            value.save_to(
                str(temporary),
                format=Types.VideoContainer.MP4,
                codec=Types.VideoCodec.AUTO
                if codec == "none"
                else Types.VideoCodec(codec),
                crf=19,
            )
            width, height = value.get_dimensions()
            metadata = {
                "width": width,
                "height": height,
                "duration": value.get_duration(),
                "fps": float(value.get_frame_rate()),
                "bit_depth": value.get_bit_depth(),
                "color_space": value.get_color_space(),
            }
        else:
            waveform = value["waveform"][0].detach().cpu().float().numpy()
            rate = value["sample_rate"]
            if waveform.shape[0] not in {1, 2}:
                raise ValueError("Audio material currently supports mono or stereo")
            layout = "mono" if waveform.shape[0] == 1 else "stereo"
            with av.open(str(temporary), "w", format="wav") as output:
                stream = output.add_stream("pcm_s16le", rate=rate)
                stream.layout = layout
                frame = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(waveform), format="fltp", layout=layout
                )
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


def create_video(
    value,
    fps=30.0,
    audio=None,
    bit_depth="auto",
    color_space="sRGB",
    codec="none",
    **kwargs,
):
    return CreateVideo.execute(value, fps, audio, bit_depth, color_space, codec).result[
        0
    ]


def read_selected(kind, directory, asset, start=0, end=0):
    project = Project(directory)
    if project.asset(asset)["kind"] != kind:
        raise ValueError("Material kind does not match this node")
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0
        or end < 0
        or (end and end <= start)
    ):
        raise ValueError("End time must be after start time")
    maximum = project.settings()["max_megapixels"]
    if kind == "video":
        video = InputImpl.VideoFromFile(
            str(project.path(asset)),
            start_time=start,
            duration=end - start if end else 0,
        )
        width, height = video.get_dimensions()
        if width * height > maximum * 1024 * 1024:
            components = video.get_components()
            scale = math.sqrt(maximum * 1024 * 1024 / (width * height))
            # Floor dimensions so native rounding cannot exceed the pixel cap.
            height, width = components.images.shape[1:3]
            width, height = max(1, int(width * scale)), max(1, int(height * scale))
            components.images = nodes.ImageScale().upscale(
                components.images, "area", width, height, "disabled"
            )[0]
            if components.alpha is not None:
                components.alpha = (
                    nodes.ImageScale()
                    .upscale(
                        components.alpha.unsqueeze(-1),
                        "area",
                        width,
                        height,
                        "disabled",
                    )[0]
                    .squeeze(-1)
                )
            video = InputImpl.VideoFromComponents(
                components,
                bit_depth=video.get_bit_depth(),
                color_space=video.get_color_space(),
            )
        return (video,)
    path = project.path(asset)
    return (
        (read_audio(path, start, end),)
        if kind == "audio"
        else (read_image(path, maximum),)
    )


def material_files(kind):
    if kind not in {"image", "video", "audio"}:
        raise ValueError("Unknown material kind")
    content_types = [kind, "video"] if kind == "audio" else [kind]
    result = []
    for root, tag in [
        (Path(folder_paths.get_input_directory()), "input"),
        (Path(folder_paths.get_output_directory()) / "materials" / kind, "output"),
    ]:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if (
                path.is_symlink()
                or not path.is_file()
                or not folder_paths.filter_files_content_types(
                    [path.name], content_types
                )
            ):
                continue
            base = (
                folder_paths.get_input_directory()
                if tag == "input"
                else folder_paths.get_output_directory()
            )
            relative = path.relative_to(base).as_posix()
            try:
                owned = material_path(relative + f" [{tag}]") == path.resolve()
            except ValueError:
                continue
            if owned:
                result.append(relative + f" [{tag}]")
    return ["", *sorted(result)]


def resolve_material(kind, file="", **kwargs):
    if kind == "audio":
        file = kwargs.get("audio", file)
    if kind == "video" and kwargs.get("images") is not None:
        value = create_video(kwargs["images"], **kwargs)
    elif kwargs.get("value") is not None:
        value = kwargs["value"]
    else:
        if not file:
            raise ValueError(f"Choose a {kind} file or connect material content")
        path = material_path(file)
        if kind == "video":
            value = InputImpl.VideoFromFile(str(path))
        elif kind == "image":
            value = nodes.LoadImage().load_image(file)[0]
        else:
            waveform, rate = load_audio(str(path))
            value = {"waveform": waveform.unsqueeze(0), "sample_rate": rate}
        output_root = Path(folder_paths.get_output_directory()).resolve()
        if not path.is_relative_to(output_root):
            stat = path.stat()
            digest = hashlib.sha256(
                f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode()
            ).hexdigest()[:12]
            destination = inside(
                output_root,
                f"materials/{kind}/{path.stem}_{digest}{path.suffix.lower()}",
            )
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(uuid.uuid4().hex + ".partial")
                try:
                    shutil.copyfile(path, temporary)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
            path = destination
        return value, path
    extension = {"image": ".png", "video": ".mp4", "audio": ".wav"}[kind]
    relative = f"materials/{kind}/{kind}_{uuid.uuid4().hex[:12]}{extension}"
    path = inside(folder_paths.get_output_directory(), relative)
    save_media(kind, value, path, codec=kwargs.get("codec", "none"))
    return value, path
