"""Task-end eviction of workspace intermediates, without clearing model caches.

ComfyUI does not expose selective local-cache eviction through CacheProvider.
Keep this version-sensitive executor integration here, away from node logic.
"""

from functools import wraps
import logging

from comfy_execution.caching import BasicCache, LRUCache, RAMPressureCache, NullCache
from execution import PromptExecutor

logger = logging.getLogger(__name__)


def release_intermediates(cache):
    if isinstance(cache, NullCache):
        return
    if not isinstance(cache, BasicCache):
        raise TypeError("Unsupported ComfyUI workspace cache implementation")
    if not cache.initialized:
        return
    removed = set()
    for node_id in cache.cache_key_set.all_node_ids():
        current = node_id
        owned = False
        while current is not None:
            kind = cache.dynprompt.get_node(current)["class_type"]
            if kind.startswith(("_TuringMaterialFresh_", "_TuringMaterialRead", "_TuringMaterialWrite")):
                owned = True
                break
            current = cache.dynprompt.get_parent_node_id(current)
        if owned:
            removed.add(cache.cache_key_set.get_data_key(node_id))
    for key in removed:
        cache.cache.pop(key, None)
        if isinstance(cache, LRUCache):
            cache.used_generation.pop(key, None)
            cache.children.pop(key, None)
        if isinstance(cache, RAMPressureCache):
            cache.timestamps.pop(key, None)
    for subcache in cache.subcaches.values():
        release_intermediates(subcache)


def install_task_cleanup():
    original = PromptExecutor.execute_async
    if getattr(original, "_turing_material_cleanup", False):
        return

    @wraps(original)
    async def execute(self, prompt, prompt_id, *args, **kwargs):
        workspace = any(n.get("class_type", "").startswith("_TuringMaterialWrite") for n in prompt.values())
        try:
            return await original(self, prompt, prompt_id, *args, **kwargs)
        finally:
            if workspace:
                for cache in self.caches.all:
                    try:
                        release_intermediates(cache)
                    except Exception:
                        # A frontend/core upgrade must not turn a successful saved
                        # result into a failure or mask the original task error.
                        # Never fall back to clearing all caches or unloading models.
                        logger.exception("Canvas intermediate cleanup failed; model caches were not globally cleared")

    execute._turing_material_cleanup = True
    PromptExecutor.execute_async = execute
