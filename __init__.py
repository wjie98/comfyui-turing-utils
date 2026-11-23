from .comfyui_turing_utils.bootstrap import bootstrap_builtin_integrations


bootstrap_builtin_integrations()

from .comfyui_turing_utils.registration import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
