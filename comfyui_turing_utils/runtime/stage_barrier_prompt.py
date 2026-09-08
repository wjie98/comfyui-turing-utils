"""Compile visual Stage Barrier hubs into independent execution paths.

ComfyUI caches and schedules whole nodes, not individual output sockets.  A
multi-input passthrough node therefore turns every connected input into a
strong dependency even when only one output is selected through a lazy node.
This prompt compiler keeps the convenient multi-port UI while replacing each
port with a private one-input/one-output node before validation and caching.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ..log import get_logger
from .stage_barrier import STAGE_BARRIER_NODE_ID, STAGE_PATH_NODE_ID


LOG = get_logger("stage")
_PROMPT_HANDLER_MARKER = "_turing_utils_stage_barrier_prompt_compiler"
_DYNAMIC_VALUE_INPUT = re.compile(r"^values\.value_(\d+)$")
_NESTED_VALUE_INPUT = re.compile(r"^value_(\d+)$")


def _input_routes(inputs: Mapping[str, Any]) -> dict[int, Any]:
    """Return explicitly supplied dynamic values indexed by visual port."""

    routes: dict[int, Any] = {}
    for name, value in inputs.items():
        match = _DYNAMIC_VALUE_INPUT.fullmatch(str(name))
        if match is not None:
            routes[int(match.group(1))] = value

    # The current frontend serializes Autogrow inputs with flattened names.
    # Accept a nested representation as well so API clients can use the same
    # compiler without depending on that frontend detail.
    nested = inputs.get("values")
    if isinstance(nested, Mapping):
        for name, value in nested.items():
            match = _NESTED_VALUE_INPUT.fullmatch(str(name))
            if match is not None:
                routes[int(match.group(1))] = value
    return routes


def _barrier_link(value: Any, barrier_ids: set[str]) -> tuple[str, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    source, output_index = value
    if not isinstance(source, str):
        return None
    if source not in barrier_ids:
        return None
    if isinstance(output_index, bool) or not isinstance(output_index, int):
        return None
    if output_index < 0:
        return None
    return source, output_index


def _collect_consumed_routes(
    value: Any,
    barrier_ids: set[str],
    consumed: dict[str, set[int]],
) -> None:
    link = _barrier_link(value, barrier_ids)
    if link is not None:
        source, output_index = link
        consumed[source].add(output_index)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _collect_consumed_routes(item, barrier_ids, consumed)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_consumed_routes(item, barrier_ids, consumed)


def _rewrite_links(value: Any, route_ids: Mapping[tuple[str, int], str]) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        source, output_index = value
        if (
            isinstance(source, str)
            and isinstance(output_index, int)
            and not isinstance(output_index, bool)
            and (source, output_index) in route_ids
        ):
            return [route_ids[(source, output_index)], 0]
    if isinstance(value, dict):
        return {name: _rewrite_links(item, route_ids) for name, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_links(item, route_ids) for item in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_links(item, route_ids) for item in value)
    return value


def _allocate_route_id(
    prompt: Mapping[str, Any],
    reserved: set[str],
    barrier_id: str,
    output_index: int,
) -> str:
    base = f"{barrier_id}.turing_stage_path.{output_index}"
    candidate = base
    collision = 0
    while candidate in prompt or candidate in reserved:
        collision += 1
        candidate = f"{base}.{collision}"
    reserved.add(candidate)
    return candidate


def compile_stage_barrier_prompt(
    prompt: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    """Return a prompt where every Stage Barrier port is an independent node.

    The transformation is deterministic for a given prompt, which preserves
    ComfyUI cache reuse between repeated runs.  It is also transactional: the
    supplied mapping and its node dictionaries are never mutated.
    """

    barrier_ids = {
        node_id
        for node_id, node in prompt.items()
        if isinstance(node, Mapping)
        and node.get("class_type") == STAGE_BARRIER_NODE_ID
    }
    if not barrier_ids:
        return dict(prompt), 0

    explicit_routes: dict[str, dict[int, Any]] = {}
    consumed = {node_id: set() for node_id in barrier_ids}
    for node_id in barrier_ids:
        node = prompt[node_id]
        inputs = node.get("inputs", {})
        explicit_routes[node_id] = (
            _input_routes(inputs) if isinstance(inputs, Mapping) else {}
        )

    for node in prompt.values():
        if not isinstance(node, Mapping):
            continue
        inputs = node.get("inputs", {})
        if isinstance(inputs, Mapping):
            _collect_consumed_routes(inputs, barrier_ids, consumed)

    route_ids: dict[tuple[str, int], str] = {}
    reserved: set[str] = set()
    for barrier_id in sorted(barrier_ids, key=str):
        indices = set(explicit_routes[barrier_id]) | consumed[barrier_id]
        for output_index in sorted(indices):
            route_ids[(barrier_id, output_index)] = _allocate_route_id(
                prompt,
                reserved,
                barrier_id,
                output_index,
            )

    compiled: dict[str, Any] = {}
    for node_id, node in prompt.items():
        if node_id in barrier_ids:
            continue
        if not isinstance(node, Mapping):
            compiled[node_id] = node
            continue
        copied = dict(node)
        inputs = node.get("inputs", {})
        if isinstance(inputs, Mapping):
            copied["inputs"] = _rewrite_links(dict(inputs), route_ids)
        compiled[node_id] = copied

    for barrier_id in sorted(barrier_ids, key=str):
        barrier = prompt[barrier_id]
        barrier_inputs = barrier.get("inputs", {})
        if not isinstance(barrier_inputs, Mapping):
            barrier_inputs = {}
        stage = barrier_inputs.get("stage", 0)
        metadata = barrier.get("_meta", {})
        base_title = (
            metadata.get("title", "Stage Barrier")
            if isinstance(metadata, Mapping)
            else "Stage Barrier"
        )
        for output_index in sorted(
            set(explicit_routes[barrier_id]) | consumed[barrier_id]
        ):
            route_inputs: dict[str, Any] = {"stage": stage}
            if output_index in explicit_routes[barrier_id]:
                route_inputs["value"] = _rewrite_links(
                    explicit_routes[barrier_id][output_index], route_ids
                )
            route_id = route_ids[(barrier_id, output_index)]
            compiled[route_id] = {
                "class_type": STAGE_PATH_NODE_ID,
                "inputs": route_inputs,
                "_meta": {
                    "title": f"{base_title} / value_{output_index}",
                    "turing_utils_stage_barrier": barrier_id,
                },
            }

    return compiled, len(route_ids)


def compile_stage_barriers_on_prompt(json_data: Any) -> Any:
    """ComfyUI ``on_prompt`` handler for route compilation."""

    if not isinstance(json_data, Mapping):
        return json_data
    prompt = json_data.get("prompt")
    if not isinstance(prompt, Mapping):
        return json_data

    compiled, route_count = compile_stage_barrier_prompt(prompt)
    if route_count == 0:
        return json_data

    result = dict(json_data)
    result["prompt"] = compiled
    LOG.info(
        "Compiled Stage Barrier hubs into %d independent lazy-compatible paths",
        route_count,
    )
    return result


def install_stage_barrier_prompt_compiler() -> bool:
    """Install the server-side prompt compiler exactly once."""

    try:
        from server import PromptServer
    except (ImportError, AttributeError):
        LOG.warning(
            "Stage Barrier route compilation is unavailable: ComfyUI's prompt "
            "server was not found"
        )
        return False

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None or not hasattr(prompt_server, "add_on_prompt_handler"):
        LOG.warning(
            "Stage Barrier route compilation is unavailable: ComfyUI's prompt "
            "server has not been initialized"
        )
        return False
    if getattr(prompt_server, _PROMPT_HANDLER_MARKER, False):
        return True

    prompt_server.add_on_prompt_handler(compile_stage_barriers_on_prompt)
    setattr(prompt_server, _PROMPT_HANDLER_MARKER, True)
    LOG.info("Enabled lazy-compatible Stage Barrier route compilation")
    return True


__all__ = [
    "compile_stage_barrier_prompt",
    "compile_stage_barriers_on_prompt",
    "install_stage_barrier_prompt_compiler",
]
