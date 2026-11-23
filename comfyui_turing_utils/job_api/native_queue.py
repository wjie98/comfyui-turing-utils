"""ComfyUI prompt validation, queue ownership, and progress event delivery."""

import asyncio
import copy
import time

from execution import SENSITIVE_EXTRA_DATA_KEYS, validate_prompt

from .errors import ValidationError


_OWNER = "turing_api_job"
_EVENTS = {
    "execution_start",
    "execution_cached",
    "progress_state",
    "execution_success",
    "execution_error",
    "execution_interrupted",
}


def _owned(item, identity):
    return item[1] == identity and item[3].get(_OWNER) == identity


class NativeQueue:
    def __init__(self, server):
        self.server = server

    async def prepare(self, identity, prompt, targets):
        data = self.server.trigger_on_prompt(
            {"prompt": prompt, "client_id": identity, "extra_data": {}}
        )
        prompt = data["prompt"]
        self.server.node_replace_manager.apply_replacements(prompt)
        valid, error, outputs, node_errors = await validate_prompt(
            identity, prompt, targets
        )
        if not valid or set(outputs) != set(targets):
            raise ValidationError(
                (error or {}).get(
                    "message", "One or more declared outputs failed validation"
                ),
                node_errors=node_errors,
            )
        return data, outputs

    def enqueue(self, identity, prepared):
        data, outputs = prepared
        extra = {
            **data.get("extra_data", {}),
            "client_id": identity,
            "create_time": int(time.time() * 1000),
            _OWNER: identity,
        }
        sensitive = {
            key: extra.pop(key) for key in SENSITIVE_EXTRA_DATA_KEYS if key in extra
        }
        queue = self.server.prompt_queue
        with queue.mutex:
            running, queued = queue.get_current_queue_volatile()
            if identity in queue.history or any(
                item[1] == identity for item in (*running, *queued)
            ):
                raise ValueError(
                    "request_id conflicts with an existing native prompt ID"
                )
            if "number" in data:
                number = float(data["number"])
            else:
                number = (
                    -self.server.number if data.get("front") else self.server.number
                )
                self.server.number += 1
            queue.put((number, identity, data["prompt"], extra, outputs, sensitive))

    def position(self, identity):
        running, queued = self.server.prompt_queue.get_current_queue_volatile()
        ordered = sorted(queued, key=lambda item: item[0])
        position = next(
            (index + 1 for index, item in enumerate(ordered) if _owned(item, identity)),
            None,
        )
        return {
            "position": position,
            "ahead": len(running) + position - 1 if position is not None else None,
        }

    def interrupt(self, identity):
        queue = self.server.prompt_queue
        with queue.mutex:
            queue.delete_queue_item(lambda item: _owned(item, identity))
            running, _ = queue.get_current_queue_volatile()
            if any(_owned(item, identity) for item in running):
                queue.interrupt_if_running(identity)

    def completion(self, identity):
        """Read history and active state atomically across the worker's handoff."""
        queue = self.server.prompt_queue
        with queue.mutex:
            record = queue.history.get(identity)
            if record is not None:
                if not _owned(record["prompt"], identity):
                    raise ValueError("Native task ownership mismatch")
                return copy.deepcopy(
                    {
                        "outputs": record["outputs"],
                        "status": record["status"],
                    }
                ), False
            running, queued = queue.get_current_queue_volatile()
            return None, any(_owned(item, identity) for item in (*running, *queued))

    def forget(self, identity):
        queue = self.server.prompt_queue
        with queue.mutex:
            record = queue.history.get(identity)
            if record and _owned(record["prompt"], identity):
                queue.delete_history_item(identity)

    def subscribe(self, listener, *, accepts):
        """Deliver task-owned events on the server loop and forward every event."""
        loop = asyncio.get_running_loop()
        original = self.server.send_sync

        def send(event, data, sid=None):
            if event in _EVENTS and accepts(sid) and data.get("prompt_id") == sid:
                graph = None
                if event == "execution_start":
                    running, _ = self.server.prompt_queue.get_current_queue_volatile()
                    item = next(
                        (entry for entry in running if _owned(entry, sid)), None
                    )
                    if item is not None:
                        graph = {
                            identity: (
                                node["class_type"],
                                node.get("_meta", {}).get("title"),
                            )
                            for identity, node in item[2].items()
                        }
                loop.call_soon_threadsafe(
                    listener, sid, event, data, time.monotonic(), graph
                )
            return original(event, data, sid)

        self.server.send_sync = send

        def unsubscribe():
            if self.server.send_sync is send:
                self.server.send_sync = original

        return unsubscribe
