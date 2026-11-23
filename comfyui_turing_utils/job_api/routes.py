"""Multipart submission and leased downloads on the existing ComfyUI server."""

import asyncio
import json
import mimetypes
import os
from pathlib import Path
import re
import time

import av
import folder_paths
from aiohttp import BodyPartReader, web
from aiohttp.helpers import content_disposition_header
from comfy.cli_args import args

from . import REQUEST_JSON_BYTES, UPLOAD_IDLE_SECONDS
from .service import Jobs, disk
from .workflow import request_fields
from .errors import ValidationError, describe
from .discovery import node_catalog, server_information


def install_routes(server):
    if getattr(server, "_turing_job_api", None) is not None:
        return
    jobs = Jobs(
        server,
        folder_paths.get_output_directory(),
        result_ttl=float(os.environ.get("TURING_API_RESULT_TTL_SECONDS", "3600")),
    )
    server._turing_job_api = jobs
    maximum = round(args.max_upload_size * 1024 * 1024)

    async def lifecycle(app):
        await jobs.start()
        try:
            yield
        finally:
            await jobs.close()

    server.app.cleanup_ctx.append(lifecycle)

    def owner(request):
        path = server.user_manager.get_request_user_filepath(
            request, "", create_dir=False
        )
        if path is None:
            raise ValueError("Unknown ComfyUI user")
        return str(path)

    def endpoint(method, path, *, requires_ready=True):
        def decorate(function):
            async def handler(request):
                if requires_ready and not jobs.ready:
                    return web.json_response(
                        {"error": jobs.error or "Job API is not ready"}, status=503
                    )
                try:
                    return await function(request)
                except TimeoutError:
                    return web.json_response({"error": "Upload timed out"}, status=408)
                except ValidationError as error:
                    return web.json_response(
                        {"error": str(error), "error_details": error.details},
                        status=400,
                    )
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    OSError,
                    av.error.FFmpegError,
                ) as error:
                    return web.json_response(
                        {
                            "error": str(error),
                            "error_details": describe(error, "request"),
                        },
                        status=400,
                    )

            getattr(server.routes, method)("/turing" + path)(handler)
            return handler

        return decorate

    def find(request):
        job = jobs.get(owner(request), request.match_info["id"])
        if job is None or job.expires is not None and job.expires <= time.time():
            raise web.HTTPNotFound(text="Unknown, expired or previous-session job")
        return job

    @endpoint("get", "/server", requires_ready=False)
    async def status(request):
        restart = any(
            route.method == "POST" and route.resource.canonical == "/manager/reboot"
            for route in server.app.router.routes()
        )
        return web.json_response(
            {
                **await disk(server_information, maximum, jobs.result_ttl),
                "session_id": jobs.session_id,
                "ready": jobs.ready,
                "error": jobs.error,
                "restart_supported": restart,
                "restart_url": "/manager/reboot" if restart else None,
            },
            status=200 if jobs.ready else 503,
        )

    @endpoint("get", "/nodes", requires_ready=False)
    async def nodes(request):
        return web.json_response(await disk(node_catalog, request.query))

    @endpoint("post", "/jobs")
    async def submit(request):
        reader = await request.multipart()
        field = await reader.next()
        if not isinstance(field, BodyPartReader) or field.name != "request":
            raise ValueError("The first multipart field must be request (JSON)")
        total, document = 0, bytearray()
        while True:
            async with asyncio.timeout(UPLOAD_IDLE_SECONDS):
                chunk = await field.read_chunk(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > min(maximum, REQUEST_JSON_BYTES):
                raise web.HTTPRequestEntityTooLarge(
                    max_size=min(maximum, REQUEST_JSON_BYTES), actual_size=total
                )
            document.extend(chunk)
        data = json.loads(document)
        workflow_field, file_fields = request_fields(data)
        job, created = await jobs.begin(owner(request), data)
        if not created:
            return web.json_response(jobs.state(job), status=202)
        uploads = {}
        expected = file_fields | {workflow_field}
        try:
            while True:
                async with asyncio.timeout(UPLOAD_IDLE_SECONDS):
                    field = await reader.next()
                if field is None:
                    break
                if (
                    not isinstance(field, BodyPartReader)
                    or field.name not in expected
                    or field.name in uploads
                    or not field.filename
                ):
                    raise ValueError("Unexpected, repeated or unnamed uploaded file")
                suffix = Path(field.filename).suffix.lower()
                if not re.fullmatch(r"\.[a-z0-9]+", suffix):
                    raise ValueError("Uploaded files require a filename extension")
                path = job.root / "uploads" / (str(len(uploads)) + suffix)
                with path.open("xb") as output:
                    received = 0
                    while True:
                        async with asyncio.timeout(UPLOAD_IDLE_SECONDS):
                            chunk = await field.read_chunk(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        received += len(chunk)
                        if total > maximum:
                            raise web.HTTPRequestEntityTooLarge(
                                max_size=maximum, actual_size=total
                            )
                        if (
                            field.name == workflow_field
                            and received > REQUEST_JSON_BYTES
                        ):
                            raise web.HTTPRequestEntityTooLarge(
                                max_size=REQUEST_JSON_BYTES, actual_size=received
                            )
                        await disk(output.write, chunk)
                uploads[field.name] = path
            if set(uploads) != expected:
                raise ValueError("One or more declared files are missing")
            workflow = await disk(
                lambda: json.loads(uploads[workflow_field].read_bytes())
            )
            await jobs.launch(job, workflow, data, uploads)
            return web.json_response(jobs.state(job), status=202)
        finally:
            job.receiver = None
            if job.runner is None:
                await jobs.discard(job)

    @endpoint("get", "/jobs/{id}")
    async def job_status(request):
        return web.json_response(jobs.state(find(request)))

    @endpoint("delete", "/jobs/{id}")
    async def delete(request):
        job = jobs.get(owner(request), request.match_info["id"])
        if job is None or await jobs.delete(job):
            return web.Response(status=204)
        return web.json_response(jobs.state(job), status=202)

    @endpoint("get", "/jobs/{id}/outputs/{output_id}")
    async def download(request):
        job = find(request)
        if job.status != "succeeded" or job.delete_requested:
            raise web.HTTPConflict(text="Job output is not available")
        if job.expires <= time.time():
            await jobs.delete(job)
            raise web.HTTPNotFound(text="Job expired")
        output = next(
            (
                item
                for item in job.outputs
                if item["id"] == request.match_info["output_id"]
            ),
            None,
        )
        if output is None:
            raise web.HTTPNotFound(text="Unknown job output")
        job.readers += 1
        try:
            response = web.StreamResponse(
                headers={
                    "Content-Type": mimetypes.guess_type(output["filename"])[0]
                    or "application/octet-stream",
                    "Content-Length": str(output["bytes"]),
                    "Content-Disposition": content_disposition_header(
                        "attachment", filename=output["filename"]
                    ),
                    "X-Content-SHA256": output["sha256"],
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                }
            )
            await response.prepare(request)
            if request.method != "HEAD":
                with output["path"].open("rb") as source:
                    while chunk := await disk(source.read, 256 * 1024):
                        await response.write(chunk)
            await response.write_eof()
            return response
        finally:
            job.readers -= 1
            if job.delete_requested:
                await jobs.delete(job)
