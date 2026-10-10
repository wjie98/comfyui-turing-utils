"""Compile single-node applications into shared, ComfyUI-owned model loaders."""

import hashlib
import json
from collections.abc import Mapping


# Application -> (loader, worker, model socket, loader defaults).
APPLICATIONS = {
    "TuringUtilsMiniMaxH3LatentUpscale": (
        "_TuringUtilsH3UpscaleLoader",
        "_TuringUtilsH3UpscaleApply",
        "upscale_model",
        {"precision": "auto"},
    ),
    "TuringUtilsSeCTrackVisualConcept": (
        "_TuringUtilsSeCLoader",
        "_TuringUtilsSeCApply",
        "model",
        {"attention": "auto"},
    ),
}


def compile_shared_loaders(prompt):
    """Keep application IDs/outputs, sharing loaders independently of runtime inputs."""
    result = dict(prompt)
    shared = {}
    for node_id, node in prompt.items():
        spec = APPLICATIONS.get(node.get("class_type"))
        if spec is None:
            continue
        loader, worker, socket, defaults = spec
        inputs = dict(node.get("inputs", {}))
        if "model_name" not in inputs:
            continue  # Let normal prompt validation report the missing input.
        loading = {"model_name": inputs.pop("model_name")}
        loading.update(
            {name: inputs.pop(name, default) for name, default in defaults.items()}
        )
        signature = json.dumps([loader, loading], sort_keys=True, separators=(",", ":"))
        if signature not in shared:
            base = (
                "_turing_loader_" + hashlib.sha256(signature.encode()).hexdigest()[:24]
            )
            shared_id = base
            suffix = 0
            while shared_id in result:
                suffix += 1
                shared_id = f"{base}_{suffix}"
            result[shared_id] = {"class_type": loader, "inputs": loading}
            shared[signature] = shared_id
        inputs[socket] = [shared[signature], 0]
        result[node_id] = {**node, "class_type": worker, "inputs": inputs}
    return result


def compile_shared_loaders_on_prompt(data):
    if not isinstance(data, Mapping) or not isinstance(data.get("prompt"), Mapping):
        return data
    return {**data, "prompt": compile_shared_loaders(data["prompt"])}


def install_shared_loader_compiler():
    from server import PromptServer

    server = getattr(PromptServer, "instance", None)
    if server is None:
        return False
    marker = "_turing_utils_shared_loader_compiler"
    if not getattr(server, marker, False):
        server.add_on_prompt_handler(compile_shared_loaders_on_prompt)
        setattr(server, marker, True)
    return True
