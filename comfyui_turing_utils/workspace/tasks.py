"""Compile a local material task from one project snapshot."""

import math
import uuid

from .compiler import compile_segment
from .execution import fresh_type
from .store import Conflict


def compile_task(project, target, revision, trims=None):
    document = project.document(card_ids={target.split(":", 1)[0]}, execution=True)
    if document["revision"] != revision:
        raise Conflict("Workflow changed; reload before running")
    for node, trim in (trims or {}).items():
        start, end = float(trim.get("start", 0)), float(trim.get("end", 0))
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end < 0
            or (end and end <= start)
        ):
            raise ValueError("Invalid trim interval")
        if node in document["selections"]:
            document["selections"][node].update(start=start, end=end)
    selection_revision = document["selections"].get(target, {}).get("revision", 0)
    run_id = str(uuid.uuid4())
    prompt = compile_segment(
        document["prompt"],
        target,
        document["selections"],
        project.directory,
        run_id,
        selection_revision,
        fresh_type,
    )
    project.begin_run(target, selection_revision, run_id)
    return {"prompt": prompt, "run_id": run_id, "target": target}
