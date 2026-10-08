"""Versioned Canvas parameters; backend node contracts remain authoritative."""

import math
from ..nodes.attention import _ATTENTION_INPUTS

VERSION = 2


def generation_values(values):
    values = dict(values)
    duration = float(values.get("duration", int(values.get("frames", 120)) / 24))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be positive and finite")
    values["duration"] = duration
    values["frames"] = max(5, round(duration * 24))
    for name in ("width", "height"):
        values[name] = max(32, round(int(values[name]) / 32) * 32)
    return values


def strategy_inputs(values):
    name = values.get("strategy", "sol" if values.get("sol") in (True, "enabled") else "disabled")
    if name == "disabled":
        return None
    if name not in _ATTENTION_INPUTS:
        raise ValueError(f"Unknown attention strategy: {name}")
    schema = _ATTENTION_INPUTS[name]()
    specs = {**schema["required"], **schema.get("optional", {})}
    result = {"strategy": name}
    policy = values.get("strategy.prefix_policy", values.get("sol.prefix_policy", "auto"))
    for key, spec in specs.items():
        if key == "model":
            continue
        default = spec[1].get("default") if len(spec) > 1 else None
        if default is None and isinstance(spec[0], list):
            default = spec[0][0] if spec[0] else ""
        path = key
        if key.startswith("sparse_reference_") or key == "manual_prefix_tokens":
            if (key == "manual_prefix_tokens") != (policy == "manual") or policy == "none":
                continue
            path = "prefix_policy." + key
        result["strategy." + path] = values.get("strategy." + path,
            values.get("sol." + key, default))
    return result
