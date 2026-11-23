"""Compile single-node applications into shared, ComfyUI-owned model loaders."""

import hashlib
import json
from collections.abc import Mapping

from ..adapters.minimax.reference_compression import migrate_reference_compression_inputs


# Application -> (loader, worker, model socket, loader defaults).
APPLICATIONS = {
    "TuringUtilsMiniMaxH3KSampler": (
        "_TuringUtilsH3UpscaleLoader",
        "_TuringUtilsH3KSamplerApply",
        "upscale_model",
        {"precision": "auto"},
    ),
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
        node_type = node.get("class_type")
        if node_type == "TuringUtilsAttentionStrategy":
            current_inputs = node.get("inputs", {})
            migrated = migrate_reference_compression_inputs(current_inputs)
            if migrated is not current_inputs:
                result[node_id] = {**node, "inputs": migrated}
        spec = APPLICATIONS.get(node_type)
        if spec is None:
            continue
        loader, worker, socket, defaults = spec
        inputs = dict(node.get("inputs", {}))
        h3_sampler = node_type == "TuringUtilsMiniMaxH3KSampler"
        if h3_sampler and (
            {"latent_second", "conditioning_second", "second_sigmas", "model_name"}
            & inputs.keys()
            or inputs.get("sigmas") == "preset"
        ):
            raise ValueError(
                "H3 sampler workflow needs migration: connect the final-size original LATENT, "
                "select single/progressive/two_pass, and replace first presets with scheduler + tail_steps."
            )
        model_key = "mode.model_name" if h3_sampler else "model_name"
        if model_key not in inputs:
            continue  # Let normal prompt validation report the missing input.
        if h3_sampler and inputs[model_key] == "none":
            continue  # The optional, lazy upscaler must not validate a nonexistent file.
        loading = {"model_name": inputs.pop(model_key)}
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
