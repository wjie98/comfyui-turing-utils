"""Compile plugin-owned workflow contracts in the native on-prompt chain."""

from collections.abc import Mapping
from contextvars import ContextVar

from ..adapters.minimax.reference_compression import (
    migrate_reference_compression_inputs,
)
from ..log import get_logger
from .shared_loaders import PromptMigrationError, compile_shared_loaders
from .stage_barrier_prompt import compile_stage_barrier_prompt


LOG = get_logger("workflow")
_MIGRATION_ERRORS: ContextVar[list[PromptMigrationError] | None] = ContextVar(
    "turing_prompt_migration_errors", default=None
)


def _migrate_vpack(prompt):
    result = dict(prompt)
    for node_id, node in prompt.items():
        if (
            isinstance(node, Mapping)
            and node.get("class_type") == "TuringUtilsAttentionStrategy"
        ):
            inputs = node.get("inputs", {})
            if isinstance(inputs, Mapping):
                migrated = migrate_reference_compression_inputs(inputs)
                if migrated is not inputs:
                    result[node_id] = {**node, "inputs": migrated}
    return result


def compile_workflow_prompt(data):
    if not isinstance(data, Mapping) or not isinstance(data.get("prompt"), Mapping):
        return data

    prompt, routes = compile_stage_barrier_prompt(data["prompt"])
    if routes:
        LOG.debug("Compiled Stage Barrier hubs into %d lazy-compatible paths", routes)
    try:
        compiled = compile_shared_loaders(_migrate_vpack(prompt))
    except PromptMigrationError as error:
        errors = _MIGRATION_ERRORS.get()
        if errors is None:
            LOG.warning("Could not migrate node %s: %s", error.node_id, error)
        else:
            errors.append(error)
        return {**data, "prompt": prompt} if routes else data
    return {**data, "prompt": compiled}


def run_prompt_handlers(server, data):
    """Run native hooks once, carrying a plugin migration failure to the caller."""
    errors = []
    token = _MIGRATION_ERRORS.set(errors)
    try:
        result = server.trigger_on_prompt(data)
    finally:
        _MIGRATION_ERRORS.reset(token)
    if errors:
        raise errors[0]
    return result


def install_workflow_prompt_compiler(server):
    if not getattr(server, "_turing_utils_workflow_prompt_compiler", False):
        server.add_on_prompt_handler(compile_workflow_prompt)
        server._turing_utils_workflow_prompt_compiler = True
