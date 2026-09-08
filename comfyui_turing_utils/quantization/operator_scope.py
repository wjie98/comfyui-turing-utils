"""Model-local comfy-kitchen dispatch for Turing Utils loaders."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps

from ..log import get_logger
from .capabilities import BACKEND_NAME, kitchen_backend_available


LOG = get_logger("dispatch")
MODEL_WRAPPER_KEY = "turing_utils_operator_backend"
CLIP_ATTACHMENT_KEY = "turing_utils_operator_backend"


@contextmanager
def use_turing_operator_backend():
    """Prefer exact Turing Utils contracts, then let Kitchen choose normally."""
    import comfy_kitchen

    with comfy_kitchen.use_backend(BACKEND_NAME):
        yield


def make_model_operator_wrapper():
    def apply_model_wrapper(executor, *args, **kwargs):
        with use_turing_operator_backend():
            return executor(*args, **kwargs)

    return apply_model_wrapper


def install_model_operator_scope(model) -> bool:
    """Scope Kitchen dispatch to one diffusion ModelPatcher."""
    if not kitchen_backend_available():
        return False
    if not callable(getattr(model, "add_wrapper_with_key", None)):
        return False

    import comfy.patcher_extension

    wrapper_type = comfy.patcher_extension.WrappersMP.APPLY_MODEL
    get_wrappers = getattr(model, "get_wrappers", None)
    if callable(get_wrappers) and get_wrappers(wrapper_type, MODEL_WRAPPER_KEY):
        return False
    model.add_wrapper_with_key(
        wrapper_type,
        MODEL_WRAPPER_KEY,
        make_model_operator_wrapper(),
    )
    LOG.info(
        "Installed operator policy: target=diffusion priority=turing_utils,kitchen"
    )
    return True


def install_clip_operator_scope(clip) -> bool:
    """Scope Kitchen dispatch to one CLIP ModelPatcher encode call."""
    if not kitchen_backend_available():
        return False
    patcher = getattr(clip, "patcher", None)
    if not all(
        callable(getattr(patcher, name, None))
        for name in (
            "add_object_patch",
            "get_attachment",
            "get_model_object",
            "set_attachments",
        )
    ):
        return False
    if patcher.get_attachment(CLIP_ATTACHMENT_KEY):
        return False

    original = patcher.get_model_object("encode_token_weights")

    @wraps(original)
    def encode_token_weights(*args, **kwargs):
        with use_turing_operator_backend():
            return original(*args, **kwargs)

    patcher.add_object_patch("encode_token_weights", encode_token_weights)
    patcher.set_attachments(CLIP_ATTACHMENT_KEY, True)
    LOG.info("Installed operator policy: target=clip priority=turing_utils,kitchen")
    return True


__all__ = [
    "CLIP_ATTACHMENT_KEY",
    "MODEL_WRAPPER_KEY",
    "install_clip_operator_scope",
    "install_model_operator_scope",
    "make_model_operator_wrapper",
    "use_turing_operator_backend",
]
