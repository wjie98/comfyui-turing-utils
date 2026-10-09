"""Project routes; actual inference is submitted through ComfyUI /prompt."""

import asyncio
import hashlib
import json
import math
from pathlib import Path
import shutil
import uuid

import av
import folder_paths
from aiohttp import web
from PIL import Image, ImageOps
from server import PromptServer

from .compiler import MATERIALS, compile_segment
from .nodes import fresh_type
from .store import Conflict, Project, inside
from . import workflow_files


def probe(path, kind):
    if kind == "text":
        path.read_text(encoding="utf-8")
        return {}
    if kind == "image":
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return {"width": image.width, "height": image.height}
    with av.open(str(path)) as container:
        streams = container.streams.video if kind == "video" else container.streams.audio
        if not streams:
            raise ValueError(f"File has no {kind} stream")
        metadata = {"duration": float(container.duration or 0) / av.time_base, "audio": bool(container.streams.audio)}
        if kind == "video":
            stream = streams[0]
            metadata.update(width=stream.width, height=stream.height, fps=float(stream.average_rate or 24))
        return metadata


def install_routes():
    server = getattr(PromptServer, "instance", None)
    if server is None or getattr(server, "_turing_material_workspace", False):
        return
    server._turing_material_workspace = True
    thumbnail_slots = asyncio.Semaphore(2)
    thumbnails = {}
    ui = Path(__file__).with_name("ui")

    def endpoint(method, path):
        def decorate(fn):
            async def wrapped(request):
                try:
                    return await fn(request)
                except Conflict as error:
                    return web.json_response({"error": str(error)}, status=409)
                except (ValueError, KeyError, TypeError, OSError, av.error.FFmpegError) as error:
                    return web.json_response({"error": str(error)}, status=400)
            getattr(server.routes, method)("/turing/workspace" + path)(wrapped)
            return wrapped
        return decorate

    @endpoint("get", "/")
    async def page(request):
        return web.FileResponse(ui / "index.html")

    @endpoint("get", "/h3-template")
    async def h3_template(request):
        return web.FileResponse(Path(__file__).resolve().parents[2] / "examples" / "h3_material_card.json")

    @endpoint("get", "/protocol")
    async def protocol(request):
        return web.json_response({"materials": MATERIALS})

    @endpoint("get", "/ui/{name}")
    async def static(request):
        name = request.match_info["name"]
        if name not in {"app.js", "style.css"}:
            raise web.HTTPNotFound()
        return web.FileResponse(ui / name)

    @endpoint("post", "/project")
    async def project(request):
        data = await request.json()
        return web.json_response(await asyncio.to_thread(lambda: Project(data["directory"]).document()))

    def library(request):
        return Path(server.user_manager.get_request_user_filepath(request, "canvas_cards", create_dir=False))

    @endpoint("post", "/templates")
    async def templates(request):
        root = library(request)
        return web.json_response({"items":await asyncio.to_thread(lambda: sorted(p.name for p in root.glob("*.json")))})

    @endpoint("post", "/template/save")
    async def save_template(request):
        data = await request.json()
        path = inside(library(request), data["name"])
        if path.parent != library(request).resolve() or path.suffix != ".json":
            raise ValueError("Use a single .json filename for a card template")
        def save_file():
            with workflow_files.LOCK:
                if path.exists():
                    raise Conflict("Template already exists; save with a new name")
                workflow_files.atomic_json(path, workflow_files.pack(data["workflow"], data["prompt"]))
        await asyncio.to_thread(save_file)
        return web.json_response({"name":path.name})

    @endpoint("post", "/card/add")
    async def add_card(request):
        data = await request.json()
        path = inside(library(request), data["name"])
        def add():
            project = Project(data["directory"])
            workflow=workflow_files.load_json(path)
            identity=workflow_files.add_instance(project.root, workflow, path.stem)
            metadata,interface=workflow_files.unpack(workflow)
            for stub,node_id in interface["stubs"].items():
                node=metadata["prompt"][node_id]
                if node["class_type"] == "TuringMaterialText":
                    asset=project.text(node["inputs"].get("text", ""),node["inputs"].get("prefix","text"))
                    project.select(f"{identity}:{stub}",{"asset":asset})
            return identity
        return web.json_response({"id":await asyncio.to_thread(add)})

    @endpoint("post", "/card/connect")
    async def connect_card(request):
        data=await request.json()
        revision=await asyncio.to_thread(lambda: workflow_files.connect(Project(data["directory"]).root,
            data["revision"],data["target"],data["port"],data.get("source"),data.get("source_slot")))
        return web.json_response({"revision":revision})

    @endpoint("post", "/card/open")
    async def open_card(request):
        data = await request.json()
        def read():
            project = Project(data["directory"])
            result = workflow_files.read_instance(project.root, data["id"])
            result["selections"] = project.selections()
            return result
        return web.json_response(await asyncio.to_thread(read))

    @endpoint("post", "/card/save")
    async def save_card(request):
        data = await request.json()
        revision = await asyncio.to_thread(lambda: workflow_files.save_instance(Project(data["directory"]).root,
            data["id"], data["workflow"], data["prompt"], data["revision"]))
        return web.json_response({"revision":revision})

    @endpoint("post", "/history")
    async def history(request):
        data = await request.json()
        result = await asyncio.to_thread(lambda: Project(data["directory"]).history(data["kind"], data.get("offset", 0), data.get("prefix", "")))
        return web.json_response({"items": result})

    @endpoint("post", "/patch")
    async def patch(request):
        data = await request.json()
        revision = await asyncio.to_thread(lambda: Project(data["directory"]).patch(data["revision"], data["changes"]))
        return web.json_response({"revision": revision})

    @endpoint("post", "/select")
    async def select(request):
        data = await request.json()
        result = await asyncio.to_thread(lambda: Project(data["directory"]).select(data["node"], data["selection"], data["revision"]))
        return web.json_response({"revision": result})

    @endpoint("post", "/text")
    async def text(request):
        data = await request.json()
        def edit():
            project = Project(data["directory"])
            asset = project.text(data["text"], data.get("prefix", "text"))
            revision = project.select(data["node"], {"asset": asset}, data["revision"])
            return {"asset": asset, "revision": revision}
        return web.json_response(await asyncio.to_thread(edit))

    @endpoint("post", "/compile")
    async def compile_request(request):
        data = await request.json()
        def build():
            project = Project(data["directory"])
            document = project.document()
            if document["revision"] != data["revision"]:
                raise Conflict("Workflow changed; reload before running")
            target = data["target"]
            revision = document["selections"].get(target, {}).get("revision", 0)
            run_id = project.begin_run(target, revision)
            prompt = compile_segment(document["prompt"], target, document["selections"],
                                     data["directory"], run_id, revision, fresh_type)
            return {"prompt": prompt, "run_id": run_id, "target": target}
        return web.json_response(await asyncio.to_thread(build))

    @endpoint("post", "/run")
    async def run(request):
        data = await request.json()
        return web.json_response(await asyncio.to_thread(lambda: Project(data["directory"]).run(data["run_id"])))

    @endpoint("post", "/metadata")
    async def metadata(request):
        data = await request.json()
        return web.json_response(await asyncio.to_thread(lambda: Project(data["directory"]).asset(data["asset"])))

    @endpoint("post", "/import")
    async def upload(request):
        data = request.query
        kind = data["kind"]
        if kind not in {"image", "video", "audio"}:
            raise ValueError("Unsupported import kind")
        project = await asyncio.to_thread(Project, data["directory"])
        reader = await request.multipart()
        field = await reader.next()
        if not field or not field.filename:
            raise ValueError("Select a file to upload")
        name = Path(field.filename)
        asset, path = project.reserve(kind, name.suffix, name.stem)
        temporary = path.with_name(path.name + ".partial")
        try:
            with temporary.open("xb") as output:
                while chunk := await field.read_chunk(1024 * 1024):
                    await asyncio.to_thread(output.write, chunk)
            info = await asyncio.to_thread(probe, temporary, kind)
            temporary.replace(path)
            await asyncio.to_thread(project.register, asset, kind, info)
        finally:
            temporary.unlink(missing_ok=True)
        return web.json_response({"asset": asset})

    @endpoint("post", "/copy")
    async def copy(request):
        data = await request.json()
        def perform():
            project = Project(data["directory"])
            source = inside(folder_paths.get_input_directory(), data["path"])
            info = probe(source, data["kind"])
            asset, path = project.reserve(data["kind"], source.suffix, source.stem)
            temporary = path.with_name(path.name + ".partial")
            try:
                shutil.copyfile(source, temporary)
                temporary.replace(path)
                project.register(asset, data["kind"], info)
            finally:
                temporary.unlink(missing_ok=True)
            return {"asset": asset}
        return web.json_response(await asyncio.to_thread(perform))

    @endpoint("get", "/asset")
    async def asset(request):
        project = await asyncio.to_thread(Project, request.query["directory"])
        path = await asyncio.to_thread(project.path, request.query["asset"])
        if request.query.get("thumbnail") == "1":
            item = await asyncio.to_thread(project.asset, request.query["asset"])
            if item["kind"] not in {"image", "video"}:
                raise ValueError("This material has no image preview")
            start = float(request.query.get("start", 0))
            if not math.isfinite(start) or start < 0:
                raise ValueError("Invalid thumbnail time")
            digest = hashlib.sha256(f"{project.root}:{item['id']}:{start:.3f}".encode()).hexdigest()
            cache = Path(folder_paths.base_path) / ".cache" / "material-previews"
            cache.mkdir(parents=True, exist_ok=True)
            preview = cache / (digest + ".jpg")
            if not preview.exists():
                def render():
                    if item["kind"] == "image":
                        with Image.open(path) as source:
                            image = ImageOps.exif_transpose(source).convert("RGB")
                    else:
                        with av.open(str(path)) as container:
                            stream = container.streams.video[0]
                            origin = float(stream.start_time * stream.time_base) if stream.start_time else 0
                            container.seek(int((origin + start) / stream.time_base), stream=stream)
                            frame = next((f for f in container.decode(video=0) if float(f.time or 0) - origin >= start), None)
                            if frame is None:
                                raise ValueError("No frame at selected time")
                            image = frame.to_image().convert("RGB")
                    image.thumbnail((512, 512))
                    temporary = preview.with_name(uuid.uuid4().hex + ".partial")
                    try:
                        image.save(temporary, format="JPEG", quality=75)
                        temporary.replace(preview)
                    finally:
                        temporary.unlink(missing_ok=True)
                async def limited():
                    async with thumbnail_slots:
                        await asyncio.to_thread(render)
                task = thumbnails.get(digest)
                if task is None:
                    task = thumbnails[digest] = asyncio.create_task(limited())
                    task.add_done_callback(lambda done: thumbnails.pop(digest, None))
                await asyncio.shield(task)
            path = preview
        return web.FileResponse(path, headers={"X-Content-Type-Options": "nosniff"})
