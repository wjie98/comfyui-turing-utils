"""Release job-owned node outputs while keeping loaded models reusable."""

from functools import wraps
from weakref import WeakKeyDictionary

import nodes
from comfy.model_patcher import is_model_patcher_output
from comfy_execution.caching import BasicCache, LRUCache, RAMPressureCache
from execution import PromptExecutor
from torch.nn import Module

from ..log import get_logger


LOG = get_logger("jobs")
_MODEL_TYPES = {
    "MODEL",
    "CLIP",
    "CLIP_VISION",
    "VAE",
    "CONTROL_NET",
    "GLIGEN",
    "STYLE_MODEL",
    "MODEL_PATCH",
    "TURING_UTILS_H3_LATENT_UPSCALE_MODEL",
    "TURING_UTILS_SEC_MODEL",
}


class JobPrompt(dict):
    """A native prompt submitted by the Job API."""


def _output_caches(cache):
    yield cache
    for child in cache.subcaches.values():
        yield from _output_caches(child)


def _contains_model(outputs):
    pending = [outputs]
    seen = set()
    while pending:
        value = pending.pop()
        if isinstance(value, (dict, list, tuple)):
            if id(value) not in seen:
                seen.add(id(value))
                pending.extend(value.values() if isinstance(value, dict) else value)
        elif isinstance(value, Module) or is_model_patcher_output(value):
            return True
    return False


def _release_job_outputs(root, prompt, previous_keys):
    for cache in _output_caches(root):
        if not cache.initialized or cache.dynprompt.original_prompt is not prompt:
            continue
        known = previous_keys.get(cache, ())
        for node_id, key in cache.cache_key_set.keys.items():
            if key in known or not cache.dynprompt.has_node(node_id):
                continue
            entry = cache.cache.get(key)
            if entry is None:
                continue
            kind = cache.dynprompt.get_node(node_id)["class_type"]
            node = nodes.NODE_CLASS_MAPPINGS.get(kind)
            output_types = getattr(node, "RETURN_TYPES", ()) or ()
            declared_model = any(
                isinstance(output, str) and output in _MODEL_TYPES
                for output in output_types
            )
            if declared_model or _contains_model(entry.outputs):
                continue
            cache.cache.pop(key)
            if isinstance(cache, LRUCache):
                cache.used_generation.pop(key, None)
                cache.children.pop(key, None)
            if isinstance(cache, RAMPressureCache):
                cache.timestamps.pop(key, None)


def install_job_cache_cleanup():
    """Add cleanup to the serial worker's execution boundary once."""
    original = PromptExecutor.execute_async
    if getattr(original, "_turing_job_cache_cleanup", False):
        return

    @wraps(original)
    async def execute(self, prompt, prompt_id, *args, **kwargs):
        if not isinstance(prompt, JobPrompt) or not isinstance(
            self.caches.outputs, BasicCache
        ):
            return await original(self, prompt, prompt_id, *args, **kwargs)

        root = self.caches.outputs
        # Snapshot keys only. Weak references let core prune old subcaches normally.
        previous_keys = WeakKeyDictionary(
            (cache, set(cache.cache)) for cache in _output_caches(root)
        )
        try:
            return await original(self, prompt, prompt_id, *args, **kwargs)
        finally:
            try:
                _release_job_outputs(root, prompt, previous_keys)
            except Exception:
                LOG.exception(
                    "Could not release node output cache for job %s", prompt_id
                )

    execute._turing_job_cache_cleanup = True
    PromptExecutor.execute_async = execute
