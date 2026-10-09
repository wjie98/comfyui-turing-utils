"""Cut an API graph at persistent materials; leave model preparation cacheable."""

import copy

from ..runtime.shared_loaders import compile_shared_loaders
from .protocol import MATERIALS


# Explicit contracts, not a guess based on a MODEL output or a class name suffix.
MODEL_PREPARATION = {
    "CheckpointLoaderSimple", "UNETLoader", "CLIPLoader", "DualCLIPLoader", "VAELoader",
    "LoraLoader", "LoraLoaderModelOnly", "CLIPSetLastLayer", "ModelSamplingSD3",
    "MiniMaxH3SigmaShift", "TuringUtilsConvRotDiffusionModelLoader",
    "TuringUtilsConvRotCLIPLoader", "TuringUtilsAttentionStrategy",
    "_TuringUtilsH3UpscaleLoader", "_TuringUtilsSeCLoader",
}


def is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


def compile_segment(prompt, target, selections, directory, run_id, revision, fresh_type):
    prompt = compile_shared_loaders(prompt)
    if target not in prompt or prompt[target]["class_type"] not in MATERIALS:
        raise ValueError("Select a material output to execute")
    compiled, visiting = {}, set()
    preparation = set()

    def preserve_configuration(key):
        if key in preparation or key not in prompt or prompt[key]["class_type"] in MATERIALS:
            return
        preparation.add(key)
        for value in prompt[key].get("inputs", {}).values():
            if is_link(value):
                preserve_configuration(value[0])

    for key, node in prompt.items():
        if node["class_type"] in MODEL_PREPARATION:
            preserve_configuration(key)

    def visit(key, destination=False):
        if key not in prompt:
            raise ValueError(f"Missing source node: {key}")
        node = prompt[key]
        kind = MATERIALS.get(node["class_type"])
        output_key = "_turing_previous_" + key if key == target and not destination else key
        if output_key in compiled:
            return output_key
        if kind and not destination:
            selected = selections.get(key, {})
            if kind == "text":
                if "text" not in selected:
                    raise ValueError(f"Text material {key} has no saved content")
                compiled[output_key] = {"class_type": "_TuringMaterialReadText", "inputs": {"text": selected["text"], "run_id": run_id}}
                return output_key
            if not selected.get("asset"):
                raise ValueError(f"Material {key} has no selected result")
            compiled[output_key] = {"class_type": "_TuringMaterialRead" + kind.title(), "inputs": {
                "directory": directory, "asset": selected["asset"],
                "start": selected.get("start", 0), "end": selected.get("end", 0),
                "include_audio": selected.get("include_audio", True), "run_id": run_id,
            }}
            return output_key
        if key in visiting:
            raise ValueError("A computation cycle must be separated by a saved material")
        visiting.add(key)
        inputs = copy.deepcopy(node.get("inputs", {}))
        if destination:
            if kind == "text":
                inputs["value"] = inputs.pop("text", None)
            if kind == "video" and "images" in inputs:
                inputs["value"] = inputs.pop("images")
            if kind != "video":
                inputs.pop("audio", None)
            inputs = {k: v for k, v in inputs.items() if k in {"value", "audio", "fps", "bit_depth", "color_space", "codec"}}
            if "value" not in inputs or not is_link(inputs["value"]):
                raise ValueError("This material has no connected computation input")
        for value in inputs.values():
            if is_link(value):
                value[0] = visit(value[0])
        visiting.remove(key)
        if destination:
            inputs.update(directory=directory, target=target, run_id=run_id, revision=revision,
                          prefix=node.get("inputs", {}).get("prefix", kind))
            class_type = "_TuringMaterialWrite" + kind.title()
        else:
            class_type = node["class_type"]
            # A model chain depending on fresh material correctly becomes fresh too.
            if key not in preparation:
                class_type = fresh_type(class_type)
        compiled[key] = {"class_type": class_type, "inputs": inputs}
        return key

    visit(target, True)
    return compiled
