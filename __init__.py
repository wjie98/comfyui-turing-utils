from .comfyui_turing_utils.bootstrap import initialize_plugin
from .comfyui_turing_utils.registration import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

initialize_plugin()

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
