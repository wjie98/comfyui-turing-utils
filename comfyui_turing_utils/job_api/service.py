"""Process-scoped API jobs; disk ownership is independent of task status."""

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import uuid

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from ..log import get_logger
from .native_queue import NativeQueue
from .workflow import prepare, result_files
from .progress import Observer, Progress
from .errors import describe

OWNER = {"owner": "turing-job-api", "format": 1}
TERMINAL = {"succeeded", "failed"}
LOG = get_logger("jobs")


async def disk(function, *args, **kwargs):
    """A cancelled handler must not remove files still owned by its I/O thread."""
    pending = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        await pending
        raise


@dataclass
class Job:
    id: str
    owner: str
    root: Path
    status: str = "receiving"
    created: float = field(default_factory=time.time)
    expires: float | None = None
    outputs: list = field(default_factory=list)
    error: str | None = None
    error_details: dict | None = None
    cleanup_error: str | None = None
    delete_requested: bool = False
    readers: int = 0
    receiver: asyncio.Task | None = None
    runner: asyncio.Task | None = None
    native_active: bool = False
    cleanup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    progress: Progress = field(default_factory=Progress)


class Jobs:
    def __init__(self, server, output_root, *, result_ttl=3600, sweep_interval=60):
        if not math.isfinite(result_ttl) or result_ttl <= 0 or sweep_interval <= 0:
            raise ValueError("Job retention and sweep interval must be positive")
        self.queue = NativeQueue(server)
        self.output_root = Path(output_root).resolve()
        self.root = self.output_root / ".turing-api-jobs"
        self.result_ttl, self.sweep_interval = result_ttl, sweep_interval
        self.session_id = uuid.uuid4().hex
        self.jobs, self.requests = {}, {}
        self.observer = Observer(self.jobs)
        self.unsubscribe = None
        self.ready, self.error = False, None
        self.lock_file = None
        self.sweeper = None

    def _start(self):
        if self.root.is_symlink():
            raise ValueError("API job root must not be a symbolic link")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        marker = self.root / ".owner.json"
        if marker.is_symlink() or (
            marker.exists() and json.loads(marker.read_text()) != OWNER
        ):
            raise ValueError("API job directory ownership mismatch")
        if not marker.exists():
            if any(self.root.iterdir()):
                raise ValueError("Refusing to adopt a nonempty API job directory")
            marker.write_text(json.dumps(OWNER))
        lock = self.root / ".lock"
        if lock.is_symlink():
            raise ValueError("API job lock must not be a symbolic link")
        self.lock_file = lock.open("a+b")
        os.set_inheritable(self.lock_file.fileno(), False)
        try:
            if os.name == "nt":
                if not self.lock_file.tell():
                    self.lock_file.write(b"\0")
                    self.lock_file.flush()
                self.lock_file.seek(0)
                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for child in self.root.iterdir():
                if child.name in {".owner.json", ".lock"}:
                    continue
                if (
                    not re.fullmatch(r"[a-f0-9]{32}", child.name)
                    or child.is_symlink()
                    or not child.is_dir()
                ):
                    raise ValueError("Unexpected entry in API job directory")
                shutil.rmtree(child)
            (self.root / self.session_id).mkdir(mode=0o700)
        except BaseException:
            self.lock_file.close()
            self.lock_file = None
            raise

    async def start(self):
        try:
            await disk(self._start)
        except (OSError, ValueError) as error:
            self.error = str(error)
            LOG.error("Startup cleanup failed: %s", error)
            return
        self.ready = True
        self.unsubscribe = self.queue.subscribe(
            self.observer.event, accepts=self.jobs.__contains__
        )
        self.sweeper = asyncio.create_task(self._sweep())

    async def close(self):
        self.ready = False
        if self.sweeper:
            self.sweeper.cancel()
            await asyncio.gather(self.sweeper, return_exceptions=True)
        pending = []
        for job in list(self.jobs.values()):
            await self.delete(job)
            if job.runner:
                pending.append(job.runner)
            if job.receiver:
                pending.append(job.receiver)
        if pending:
            _, running = await asyncio.wait(pending, timeout=5)
            if running or any(job.native_active for job in self.jobs.values()):
                # Keep the directory lock until process exit if native work is
                # still stopping. A second process must not erase its files.
                return
        if self.lock_file:
            self.lock_file.close()
            self.lock_file = None
        if self.unsubscribe is not None:
            self.unsubscribe()
            self.unsubscribe = None

    async def begin(self, owner, request):
        if not self.ready:
            raise RuntimeError(self.error or "Job API is not ready")
        supplied = request.get("session_id")
        if supplied is not None and supplied != self.session_id:
            raise ValueError("Server restarted; submit a new task")
        identity = (
            str(uuid.UUID(request["request_id"]))
            if request.get("request_id")
            else str(uuid.uuid4())
        )
        signature = (
            owner,
            hashlib.sha256(
                json.dumps(request, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
        )
        if identity in self.requests:
            if self.requests[identity] != signature:
                raise ValueError("request_id was already used for another request")
            if identity not in self.jobs:
                raise ValueError("This request was deleted; use a new request_id")
            return self.jobs[identity], False
        root = self.root / self.session_id / uuid.UUID(identity).hex
        job = Job(identity, owner, root)
        job.receiver = asyncio.current_task()
        self.requests[identity], self.jobs[identity] = signature, job
        try:
            await disk(root.mkdir, mode=0o700)
            await disk((root / "uploads").mkdir, mode=0o700)
        except BaseException:
            await self.discard(job)
            raise
        return job, True

    def get(self, owner, identity):
        job = self.jobs.get(identity)
        return job if job and job.owner == owner else None

    def state(self, job):
        status = (
            "running"
            if job.status in TERMINAL and job.progress.finished is None
            else job.status
        )
        return {
            "id": job.id,
            "session_id": self.session_id,
            "status": "deleting" if job.delete_requested else status,
            "created": job.created,
            "expires": job.expires,
            "error": job.error,
            "error_details": job.error_details,
            "cleanup_error": job.cleanup_error,
            "outputs": [
                {k: v for k, v in item.items() if k != "path"} for item in job.outputs
            ],
            "queue": self.queue.position(job.id),
            **job.progress.snapshot(self.observer.history),
        }

    async def launch(self, job, workflow, request, uploads):
        job.status = "preparing"
        job.progress.transition("preparing")
        prompt, signature = await disk(prepare, workflow, request, uploads, job.root)
        targets = list(
            dict.fromkeys(entry["node_id"] for entry in request["outputs"].values())
        )
        prepared = await self.queue.prepare(job.id, prompt, targets)
        job.progress.signature, job.progress.targets = (
            (job.owner, signature),
            set(targets),
        )
        if job.delete_requested:
            return
        self.queue.enqueue(job.id, prepared)
        job.status, job.receiver, job.native_active = "queued", None, True
        job.progress.transition("queued")
        job.runner = asyncio.create_task(self._execute(job, request["outputs"]))

    async def _execute(self, job, outputs):
        try:
            while True:
                record, active = self.queue.completion(job.id)
                if record is not None:
                    if record["status"]["status_str"] != "success":
                        event, details = next(
                            (
                                (event, data)
                                for event, data in reversed(
                                    record["status"]["messages"]
                                )
                                if event in {"execution_error", "execution_interrupted"}
                            ),
                            ("execution_error", {}),
                        )
                        job.error_details = {
                            "code": "execution_interrupted"
                            if event == "execution_interrupted"
                            else "execution_failed",
                            "phase": "executing",
                            "node_id": details.get("node_id"),
                            "class_type": details.get("node_type"),
                            "exception_type": details.get("exception_type"),
                            "message": details.get("exception_message")
                            or (
                                "Workflow execution was interrupted"
                                if event == "execution_interrupted"
                                else "Workflow execution failed"
                            ),
                        }
                        raise ValueError(
                            details.get("exception_message")
                            or "Workflow execution failed or was interrupted"
                        )
                    break
                if not active:
                    raise ValueError(
                        "Native task was cancelled or removed from the queue"
                    )
                await asyncio.sleep(0.05)
            job.native_active = False
            if not job.delete_requested:
                job.progress.transition("finalizing")
                job.outputs = await disk(self._results, job, outputs, record)
                job.progress.completed_targets.update(job.progress.targets)
        except asyncio.CancelledError:
            job.status, job.error = "failed", "Job worker stopped"
            self.queue.interrupt(job.id)
            raise
        except Exception as error:
            job.native_active = False
            job.status, job.error = "failed", str(error)
            if job.error_details is None:
                job.error_details = describe(error, job.progress.phase)
        finally:
            if not job.native_active:
                await self._finish(job)

    async def _finish(self, job):
        job.progress.transition("cleanup")
        job.expires = time.time() + self.result_ttl
        try:
            self.queue.forget(job.id)
            if job.delete_requested:
                await self.discard(job)
            else:
                await disk(self._clear_work, job)
        except OSError as error:
            job.cleanup_error = str(error)
        finally:
            if job.status != "failed":
                job.status = "succeeded"
            job.progress.finish("deleting" if job.delete_requested else job.status)
            if not job.delete_requested:
                self.observer.remember(job)

    def _results(self, job, outputs, record):
        destination = job.root / "outputs"
        destination.mkdir()
        result = []
        counts = {}
        for name, source, ui in result_files(record, outputs, job.root):
            index = counts.get(name, 0)
            counts[name] = index + 1
            identity = uuid.uuid4().hex
            text = ui.get("text") if isinstance(ui, dict) else None
            is_text = isinstance(text, list) and all(
                isinstance(item, str) for item in text
            )
            suffix = source.suffix if source else ".txt" if is_text else ".json"
            path = destination / (identity + suffix)
            digest = hashlib.sha256()
            with path.open("xb") as output:
                if source:
                    with source.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            output.write(chunk)
                            digest.update(chunk)
                else:
                    chunk = (
                        "\n".join(text)
                        if is_text
                        else json.dumps(ui, ensure_ascii=False)
                    ).encode("utf-8")
                    output.write(chunk)
                    digest.update(chunk)
            filename = (
                re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(". ") or "result"
            )
            filename = filename.encode("utf-8")[:160].decode("utf-8", errors="ignore")
            extension = suffix.lower()
            kind = (
                "image"
                if extension in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
                else "video"
                if extension in {".mp4", ".webm", ".mov", ".mkv"}
                else "audio"
                if extension in {".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a"}
                else "text"
                if is_text
                else "json"
                if source is None
                else "file"
            )
            result.append(
                {
                    "id": identity,
                    "name": name,
                    "index": index,
                    "kind": kind,
                    "filename": f"{filename}_{len(result) + 1}{suffix}",
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                    "path": path,
                    "url": f"/turing/jobs/{job.id}/outputs/{identity}",
                }
            )
        return result

    @staticmethod
    def _clear_work(job):
        for name in ("uploads", "work"):
            path = job.root / name
            if path.exists():
                shutil.rmtree(path)
        if job.status == "failed":
            shutil.rmtree(job.root)

    async def delete(self, job):
        first = not job.delete_requested
        job.delete_requested = True
        if job.receiver and not job.receiver.done():
            if first:
                job.receiver.cancel()
            return False
        if job.native_active or job.runner and not job.runner.done():
            self.queue.interrupt(job.id)
            return False
        if job.readers:
            return False
        return await self.discard(job)

    async def discard(self, job):
        async with job.cleanup_lock:
            try:
                if job.root.exists():
                    await disk(shutil.rmtree, job.root)
            except OSError as error:
                job.delete_requested, job.cleanup_error = True, str(error)
                return False
            self.jobs.pop(job.id, None)
            return True

    async def _sweep(self):
        while True:
            await asyncio.sleep(self.sweep_interval)
            for job in list(self.jobs.values()):
                if job.delete_requested or (
                    job.expires is not None and job.expires <= time.time()
                ):
                    await self.delete(job)
                elif job.cleanup_error and job.status in TERMINAL:
                    try:
                        await disk(self._clear_work, job)
                        job.cleanup_error = None
                    except OSError as error:
                        job.cleanup_error = str(error)
