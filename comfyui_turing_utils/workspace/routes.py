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
from .templates import material_template


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

    @endpoint("get", "/new-template")
    async def new_template(request):
        return web.json_response(material_template()[0])

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

    @endpoint("post", "/projects")
    async def projects(request):
        data = await request.json()
        root = Path(folder_paths.get_output_directory()).resolve()
        relative = data.get("path", "")
        parent = inside(root, relative) if relative else root
        def listing():
            return [{"name":p.name, "path":p.relative_to(root).as_posix(), "project":(p/"canvas.json").is_file()}
                    for p in sorted(parent.iterdir()) if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")]
        return web.json_response({"items":await asyncio.to_thread(listing), "path":relative})

    def template_path(request, name):
        root = library(request)
        path = inside(root, name)
        if path.parent != root.resolve() or path.suffix != ".json":
            raise ValueError("Use a single .json filename for a card template")
        return path

    @endpoint("post", "/template/open")
    async def open_template(request):
        data = await request.json()
        raw = await asyncio.to_thread(template_path(request, data["name"]).read_bytes)
        return web.json_response({"workflow":json.loads(raw), "revision":hashlib.sha256(raw).hexdigest()})

    @endpoint("post", "/templates")
    async def templates(request):
        root = library(request)
        return web.json_response({"items":await asyncio.to_thread(lambda: sorted(p.name for p in root.glob("*.json")))})

    @endpoint("post", "/template/save")
    async def save_template(request):
        data = await request.json()
        path = template_path(request, data["name"])
        def save_file():
            with workflow_files.LOCK:
                if path.exists():
                    if data.get("revision") != hashlib.sha256(path.read_bytes()).hexdigest():
                        raise Conflict("Template exists or changed; open it before replacing it")
                workflow_files.atomic_json(path, workflow_files.pack(data["workflow"], data["prompt"]))
                return hashlib.sha256(path.read_bytes()).hexdigest()
        revision = await asyncio.to_thread(save_file)
        return web.json_response({"name":path.name, "revision":revision})

    @endpoint("post", "/card/add")
    async def add_card(request):
        data = await request.json()
        def add():
            project = Project(data["directory"])
            kind = data.get("kind")
            if kind:
                if kind not in {"image", "video", "audio", "text"}:
                    raise ValueError("Unknown material type")
                source, prompt = material_template((kind,))
                workflow, title = workflow_files.pack(source, prompt), kind.title()
            else:
                path = template_path(request, data["name"])
                workflow, title = workflow_files.load_json(path), path.stem
            metadata,interface=workflow_files.unpack(workflow)
            selections = {}
            for stub,node_id in interface["stubs"].items():
                node=metadata["prompt"][node_id]
                if node["class_type"] == "TuringMaterialText":
                    selections[stub] = {"text":node["inputs"].get("text", "")}
                elif node["inputs"].get("file"):
                    relative, base = folder_paths.annotated_filepath(node["inputs"]["file"])
                    base = base or folder_paths.get_input_directory()
                    if Path(base).resolve() not in {Path(folder_paths.get_input_directory()).resolve(),Path(folder_paths.get_output_directory()).resolve()}:
                        raise ValueError("Material must belong to this instance's input or output")
                    source = inside(base, relative)
                    kind = MATERIALS[node["class_type"]]
                    info = probe(source,kind)
                    asset,destination = project.reserve(kind,source.suffix,source.stem)
                    shutil.copyfile(source,destination)
                    project.register(asset,kind,info)
                    selections[stub] = {"asset":asset}
            identity=workflow_files.add_instance(project.root, workflow, title)
            for stub,selection in selections.items():
                project.select(f"{identity}:{stub}",selection)
            if "x" in data and "y" in data:
                project.patch(project.document()["revision"], [{"type":"position", "id":identity, "x":data["x"], "y":data["y"]}])
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
            revision = project.select(data["node"], {"text": data["text"]}, data["revision"])
            return {"text": data["text"], "revision": revision}
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
            for node, trim in data.get("trims", {}).items():
                start, end = float(trim.get("start", 0)), float(trim.get("end", 0))
                if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < 0 or (end and end <= start):
                    raise ValueError("Invalid trim interval")
                if node in document["selections"]:
                    document["selections"][node].update(start=start, end=end)
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
