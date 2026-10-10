"""Fresh execution adapters; model preparation remains owned by ComfyUI caches."""

import copy
import nodes
from comfy_api.latest import io
from ..nodes import INTERNAL_NODE_NOTE


def fresh_type(name):
    """Isolated derived classes: never mutate third-party classes or the executor."""
    original = nodes.NODE_CLASS_MAPPINGS[name]
    key = "_TuringMaterialFresh_" + name
    if key not in nodes.NODE_CLASS_MAPPINGS:
        fingerprint = classmethod(lambda cls, **kwargs: float("nan"))
        attributes = {
            "IS_CHANGED": fingerprint,
            "DEV_ONLY": True,
            "CATEGORY": "",
            "DESCRIPTION": INTERNAL_NODE_NOTE,
        }
        if issubclass(original, io.ComfyNode):
            attributes["fingerprint_inputs"] = fingerprint

            def schema(cls):
                result = copy.deepcopy(original.define_schema())
                result.node_id = key
                result.display_name = f"{result.display_name or name} Fresh (Internal)"
                result.description = INTERNAL_NODE_NOTE + (result.description or "")
                result.category = ""
                result.is_dev_only = True
                return result

            attributes["define_schema"] = classmethod(schema)
        nodes.NODE_CLASS_MAPPINGS[key] = type(key, (original,), attributes)
        nodes.NODE_DISPLAY_NAME_MAPPINGS[key] = f"{name} Fresh (Internal)"
    return key
