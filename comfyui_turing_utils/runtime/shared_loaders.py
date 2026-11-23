"""Compile single-node applications into shared, ComfyUI-owned model loaders."""

import hashlib
import json
from collections.abc import Mapping

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


class PromptMigrationError(ValueError):
    def __init__(self, node_id, class_type, message):
        super().__init__(message)
        self.node_id = node_id
        self.class_type = class_type


def compile_shared_loaders(prompt):
    """Keep application IDs/outputs, sharing loaders independently of runtime inputs."""
    result = dict(prompt)
    shared = {}
    for node_id, node in prompt.items():
        if not isinstance(node, Mapping):
            continue
        node_type = node.get("class_type")
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
            raise PromptMigrationError(
                node_id,
                node_type,
                "H3 sampler workflow needs migration: connect the final-size original LATENT, "
                "select single/progressive/two_pass, and replace first presets with scheduler + tail_steps.",
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
