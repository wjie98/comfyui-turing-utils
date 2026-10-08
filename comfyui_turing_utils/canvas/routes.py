"""Canvas project API. Generation still runs exclusively in ComfyUI's queue."""

import asyncio
import json
import os
import time
import uuid
import hashlib
from pathlib import Path
import numpy as np
import av
import folder_paths
from PIL import Image, ImageOps, UnidentifiedImageError

from aiohttp import web
import execution
from server import PromptServer

from ..nodes.multimodal_chat import MultimodalPromptChat
from .compiler import compile_task
from .graph import ASSETS, H3, PUBLIC, material_inputs, plan, validate_canvas, image_ports, reference_ports
from .media import read_material
from .store import Project, atomic_json, local_source, probe_import, contained, remove_empty_directory
from .schema import VERSION, generation_values


def context(data, create=True):
    nodes, settings = validate_canvas(data["graph"])
    project = Project(settings["work_directory"], create=create)
    if create:
        project.configure_cache(settings["cache_directory"])
    return nodes, settings, project


def guard_prompt(data):
    # on_prompt exceptions are swallowed by core: return an invalid prompt instead.
    prompt = data.get("prompt", {})
    workflow = data.get("extra_data", {}).get("extra_pnginfo", {}).get("workflow", {})
    types = {v.get("class_type") for v in prompt.values()}
    types.update(n.get("type") for n in workflow.get("nodes", []))
    if types & PUBLIC:
        return {**data, "prompt": {"canvas_error": {
            "class_type": "Canvas mode: use card buttons; ordinary or mixed workflow execution is disabled",
            "inputs": {}}}}
    return data


def install_canvas_routes():
    server = getattr(PromptServer, "instance", None)
    if server is None or getattr(server, "_turing_canvas_installed", False):
        return
    server._turing_canvas_installed = True
    server.add_on_prompt_handler(guard_prompt)

    def endpoint(method, path):
        def decorate(fn):
            async def wrapped(request):
                try:
                    return await fn(request)
                except (ValueError, KeyError, TypeError, OSError, av.error.FFmpegError, UnidentifiedImageError) as error:
                    return web.json_response({"error": str(error)}, status=400)
            getattr(server.routes, method)("/turing/canvas/" + path)(wrapped)
            return wrapped
        return decorate

    @endpoint("post", "state")
    async def state(request):
        data = await request.json()
        _, _, project = context(data, create=False)
        tasks = {k: {"asset": v["asset"], "signature": v["signature"]} for k, v in project.state()["tasks"].items()}
        return web.json_response({"state": {"tasks": tasks}, "plan": plan(data["graph"], project)})

    @endpoint("post", "select")
    async def select(request):
        data = await request.json()
        nodes, _, project = context(data)
        node = nodes[str(data["task"])]
        if node["type"] != H3:
            raise ValueError("Select generation history on an H3 node")
        prefix = node["values"].get("filename_prefix", "h3")
        if data["asset"] not in {a["id"] for a in project.history("video", prefix)}:
            raise ValueError("Result is not in this filename prefix")
        metadata = project.asset(data["asset"])["metadata"]
        project.publish(data["task"], data["asset"], metadata["signature"], metadata["snapshot"], generated=False)
        return web.json_response({"ok": True})

    @endpoint("post", "history")
    async def history(request):
        data = await request.json()
        nodes, _, project = context(data)
        node = nodes[str(data["task"])]
        kind = ASSETS.get(node["type"], "video")
        prefix = node["values"].get("filename_prefix", "h3") if node["type"] == H3 else None
        return web.json_response({"items": project.history(kind, prefix)})

    @endpoint("post", "directories")
    async def directories(request):
        data = await request.json()
        root = Path(folder_paths.get_output_directory()).resolve()
        relative = data.get("path", "")
        directory = contained(root, relative) if relative else root
        children = await asyncio.to_thread(lambda: sorted(p.name for p in directory.iterdir() if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")))
        return web.json_response({"path": directory.relative_to(root).as_posix(), "directories": children})

    @endpoint("post", "remove-directory")
    async def remove_directory(request):
        data = await request.json()
        await asyncio.to_thread(remove_empty_directory, data["path"])
        return web.json_response({"ok": True})

    @endpoint("post", "metadata")
    async def metadata(request):
        data = await request.json()
        _, _, project = context(data, create=False)
        item = await asyncio.to_thread(project.asset, data["asset"])
        return web.json_response(item["metadata"])

    @endpoint("post", "restore")
    async def restore(request):
        data = await request.json()
        nodes, _, project = context(data, create=False)
        task = str(data["task"])
        if nodes[task]["type"] == H3:
            state = project.state()["tasks"].get(task, {})
            records = state.get("history", [])
            if not records:
                raise ValueError("This node has no successful generation to restore")
            snapshot = project.asset(state.get("last_generated", records[-1]["asset"]))["metadata"]["snapshot"]
            if snapshot.get("template", 1) > VERSION:
                raise ValueError("Result uses a newer Canvas schema")
            values = generation_values(snapshot["values"])
        else:
            path = contained(project.root, "parameters.json")
            if not path.exists():
                raise ValueError("Save the Canvas project once to create a parameter checkpoint")
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            if checkpoint.get("schema_version", 1) > VERSION:
                raise ValueError("Checkpoint uses a newer Canvas schema")
            values = checkpoint["nodes"][task]["values"]
        # Restoring root settings must not silently switch the project being edited.
        return web.json_response({"values": {k: v for k, v in values.items() if k not in {"work_directory", "cache_directory", "seed", "frames"}}})

    @endpoint("post", "save")
    async def save(request):
        data = await request.json()
        _, _, project = context(data)
        atomic_json(project.root / "canvas.json", data["workflow"])
        atomic_json(project.root / "parameters.json", {"schema_version": VERSION,
            "nodes": {str(n["id"]): {"type": n["type"], "values": n["values"]} for n in data["graph"]["nodes"]}})
        return web.json_response({"ok": True})

    @endpoint("post", "load")
    async def load(request):
        data = await request.json()
        _, _, project = context(data)
        return web.json_response(json.loads(contained(project.root, "canvas.json").read_text(encoding="utf-8")))

    @endpoint("post", "import")
    async def import_local(request):
        data = await request.json()
        nodes, settings, project = context(data)
        if settings["import_mode"] != "local_copy":
            raise ValueError("Enable local_copy in Canvas Settings first")
        node = nodes[str(data["task"])]
        kind = ASSETS[node["type"]]
        source = local_source(node["values"]["local_path"])
        asset = await asyncio.to_thread(project.copy, source, kind)
        return web.json_response(asset)

    @endpoint("post", "upload")
    async def upload(request):
        # Multipart is streamed to disk; no base64 or whole-video request buffer.
        reader = await request.multipart()
        first = await reader.next()
        if first is None or first.name != "canvas":
            raise ValueError("Canvas metadata must be the first multipart field")
        metadata = bytearray()
        while chunk := await first.read_chunk():
            metadata.extend(chunk)
            if len(metadata) > 2 * 1024 * 1024:
                raise ValueError("Canvas metadata is too large")
        data = json.loads(metadata)
        nodes, settings, project = context(data)
        if settings["import_mode"] != "browser_upload":
            raise ValueError("Local-copy mode cannot obtain the server path from a browser file drop; enter local_path and click Load")
        node = nodes[str(data["task"])]
        field = await reader.next()
        if field is None or not field.filename:
            raise ValueError("Missing uploaded file")
        asset_id, path = project.reserve(Path(field.filename).suffix, kind=ASSETS[node["type"]], name=Path(field.filename).name)
        temporary = path.with_suffix(path.suffix + ".partial")
        try:
            with temporary.open("xb") as file:
                while chunk := await field.read_chunk(1024 * 1024):
                    await asyncio.to_thread(file.write, chunk)
            os.replace(temporary, path)
            kind = ASSETS[node["type"]]
            info = await asyncio.to_thread(probe_import, path, kind)
            asset = project.register(asset_id, path, kind, Path(field.filename).name, info)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)
        return web.json_response(asset)

    @endpoint("get", "asset")
    async def asset(request):
        project = Project(request.query["directory"])
        path = project.path(request.query["id"])
        kind = project.asset(request.query["id"])["kind"]
        if request.query.get("preview") == "1" and kind in {"mask", "image", "video"}:
            preview_path = project.cache() / (hashlib.sha256(request.query["id"].encode()).hexdigest() + "-preview.jpg")
            if not preview_path.exists():
                def thumbnail():
                    if kind == "mask":
                        mask = np.load(path, mmap_mode="r", allow_pickle=False)[0]
                        image = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)).convert("RGB")
                    elif kind == "video":
                        with av.open(str(path)) as container:
                            frame = next(container.decode(video=0), None)
                            if frame is None:
                                raise ValueError("Video has no decodable frames")
                            image = frame.to_image().convert("RGB")
                    else:
                        with Image.open(path) as source:
                            image = ImageOps.exif_transpose(source).convert("RGB")
                    image.thumbnail((512, 512))
                    temporary = preview_path.with_name(uuid.uuid4().hex + ".partial")
                    try:
                        image.save(temporary, format="JPEG", quality=75)
                        os.replace(temporary, preview_path)
                    finally:
                        temporary.unlink(missing_ok=True)
                await asyncio.to_thread(thumbnail)
            path = preview_path
        return web.FileResponse(path, headers={"X-Content-Type-Options": "nosniff"})

    @endpoint("post", "run")
    async def run(request):
        data = await request.json()
        _, _, project = context(data)
        prompt = compile_task(data["graph"], data["task"], project)
        prompt_id = str(uuid.uuid4())
        valid = await execution.validate_prompt(prompt_id, prompt, ["publish"])
        if not valid[0]:
            return web.json_response({"error": valid[1], "node_errors": valid[3]}, status=400)
        if data.get("validate_only"):
            return web.json_response({"valid": True, "nodes": list(prompt)})
        number = server.number
        server.number += 1
        extra = {"client_id": data.get("client_id"), "create_time": int(time.time() * 1000),
                 "turing_canvas": {"task": data["task"], "directory": project.directory}}
        server.prompt_queue.put((number, prompt_id, prompt, extra, valid[2], {}))
        return web.json_response({"prompt_id": prompt_id})

    @endpoint("post", "enhance")
    async def enhance(request):
        data = await request.json()
        nodes, settings, project = context(data)
        node = nodes[str(data["task"])]
        if node["type"] != H3:
            raise ValueError("Only H3 cards support prompt enhancement")
        materials = material_inputs(node, nodes, project.state(), project)
        images, videos, keyframes = {}, {}, {}
        for port in (*image_ports(materials), *reference_ports(materials, "video"), "prefix", "target", "first_frame", "last_frame"):
            if port not in materials:
                continue
            ref = {**materials[port], "include_audio": False}
            pixels, _, _ = await asyncio.to_thread(read_material, project, ref, 384, 384, 240)
            if port in ("first_frame", "last_frame"):
                keyframes[port] = pixels[:1]
            elif port in image_ports(materials):
                images[f"image_{len(images)}"] = pixels
            else:
                videos[f"video_{len(videos)}"] = pixels
        key_env = settings.get("chat_api_key_env", "")
        if key_env and not key_env.replace("_", "").isalnum():
            raise ValueError("chat_api_key_env must be an environment variable name")
        result = await MultimodalPromptChat.execute(prompt=node["values"].get("user_prompt", ""),
            system_prompt=settings.get("chat_system_prompt", ""), base_url=settings["chat_url"],
            model=settings["chat_model"], api_key=("$" + key_env) if key_env else "",
            images=images, videos=videos, **keyframes)
        return web.json_response({"model_prompt": result.result[0]})
