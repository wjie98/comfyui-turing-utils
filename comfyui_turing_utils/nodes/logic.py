"""Small workflow-control helpers."""

from __future__ import annotations

import re

import torch
from comfy_api.latest import io

from ..log import get_logger
from ..runtime.stage_barrier import STAGE_BARRIER_NODE_ID, STAGE_PATH_NODE_ID


LOG = get_logger("stage")
_MISSING = object()
_MAX_STAGE_BARRIER_VALUES = 100
_DYNAMIC_VALUE_SUFFIX = re.compile(r"(\d+)$")


def is_value_present(value=None) -> bool:
    """Return whether an optional workflow value exists and is non-empty.

    Presence is intentionally different from Python truthiness: connected
    scalar values such as ``0`` and ``False`` are valid values.  Only a missing
    value, ``None``, or an object with no elements is considered absent.
    """
    if value is None:
        return False
    if torch.is_tensor(value):
        return value.numel() > 0
    try:
        return len(value) > 0
    except (TypeError, AttributeError):
        return True


class IsInputPresent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsIsInputPresent",
            display_name="Is Input Present",
            category="Turing Utils/logic",
            description=(
                "Returns true when the optional input is connected and non-empty. "
                "The value output forwards the primary input when present, otherwise "
                "it lazily evaluates and forwards the optional fallback. Zero and "
                "false scalar values still count as present."
            ),
            inputs=[
                io.AnyType.Input(
                    "value",
                    optional=True,
                    tooltip=(
                        "Connect any value. An unconnected input, None, an empty "
                        "string/container, or a zero-element tensor returns false."
                    ),
                ),
                io.AnyType.Input(
                    "fallback",
                    optional=True,
                    lazy=True,
                    tooltip=(
                        "Returned only when value is absent or empty. Its upstream "
                        "branch is evaluated lazily."
                    ),
                ),
            ],
            outputs=[
                io.Boolean.Output(display_name="present"),
                io.AnyType.Output(display_name="value"),
            ],
        )

    @classmethod
    def check_lazy_status(cls, value=None, fallback=_MISSING):
        if is_value_present(value) or fallback is _MISSING:
            return None
        if fallback is None:
            return ["fallback"]

    @classmethod
    def execute(cls, value=None, fallback=_MISSING) -> io.NodeOutput:
        present = is_value_present(value)
        selected = value if present else fallback
        return io.NodeOutput(
            present,
            None if selected is _MISSING else selected,
        )


class LazyIfElse(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsLazyIfElse",
            display_name="Lazy If / Else",
            category="Turing Utils/logic",
            description=(
                "Returns the selected branch and asks ComfyUI to evaluate only that "
                "lazy input. An unselected branch is skipped unless another output "
                "in the workflow also requires it."
            ),
            search_aliases=["if", "else", "switch", "branch", "lazy"],
            inputs=[
                io.Boolean.Input("condition"),
                io.AnyType.Input(
                    "on_true",
                    lazy=True,
                    optional=True,
                    tooltip="Evaluated only when condition is true.",
                ),
                io.AnyType.Input(
                    "on_false",
                    lazy=True,
                    optional=True,
                    tooltip="Evaluated only when condition is false.",
                ),
            ],
            outputs=[io.AnyType.Output(display_name="value")],
        )

    @classmethod
    def check_lazy_status(
        cls, condition, on_true=_MISSING, on_false=_MISSING
    ):
        selected = on_true if condition else on_false
        if selected is _MISSING:
            return None
        if condition and selected is None:
            return ["on_true"]
        if not condition and selected is None:
            return ["on_false"]

    @classmethod
    def execute(
        cls, condition, on_true=_MISSING, on_false=_MISSING
    ) -> io.NodeOutput:
        selected = on_true if condition else on_false
        return io.NodeOutput(None if selected is _MISSING else selected)


class StageBarrier(io.ComfyNode):
    """Cacheable arbitrary-value rendezvous with dependency-first scheduling."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=STAGE_BARRIER_NODE_ID,
            display_name="Stage Barrier",
            category="Turing Utils/logic",
            description=(
                "Tag independent arbitrary-value paths for dependency-first phase "
                "ordering. Each visual input/output pair is compiled into its own "
                "cache and execution unit, so unselected lazy branches stay skipped. "
                "Stage is a reusable phase label: barriers with the same inferred "
                "round and stage rendezvous before downstream work is released, and "
                "a dependency whose stage decreases automatically starts a new round."
            ),
            search_aliases=["barrier", "stage", "rendezvous", "execution order"],
            inputs=[
                io.Int.Input(
                    "stage",
                    default=0,
                    min=0,
                    max=2**31 - 1,
                    step=1,
                    socketless=True,
                    tooltip=(
                        "Phase label within an automatically inferred dependency round. "
                        "Use values such as 0=prepare, 1=sample, 2=decode and reuse "
                        "them in dependent rounds. This must remain a widget so the "
                        "scheduler can plan it before executing upstream nodes."
                    ),
                ),
                io.Autogrow.Input(
                    "values",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.AnyType.Input("value"),
                        prefix="value_",
                        min=0,
                        max=_MAX_STAGE_BARRIER_VALUES,
                    ),
                    tooltip=(
                        "Connect any number and mixture of ComfyUI values. Each input "
                        "is forwarded to the matching output without copying it."
                    ),
                ),
            ],
            outputs=[
                io.AnyType.Output(f"value_{index}")
                for index in range(_MAX_STAGE_BARRIER_VALUES)
            ],
        )

    @classmethod
    def execute(cls, stage, values=None) -> io.NodeOutput:
        stage = int(stage)
        if stage < 0:
            raise ValueError("Stage Barrier stage must be greater than or equal to zero")

        outputs = [None] * _MAX_STAGE_BARRIER_VALUES
        connected = 0
        for name, value in (values or {}).items():
            match = _DYNAMIC_VALUE_SUFFIX.search(str(name))
            if match is None:
                raise ValueError(f"Invalid Stage Barrier dynamic input name: {name}")
            index = int(match.group(1))
            if index >= _MAX_STAGE_BARRIER_VALUES:
                raise ValueError(
                    f"Stage Barrier supports at most {_MAX_STAGE_BARRIER_VALUES} values"
                )
            outputs[index] = value
            connected += 1

        LOG.info("Stage Barrier reached: stage=%d values=%d", stage, connected)
        return io.NodeOutput(*outputs)


class StagePath(io.ComfyNode):
    """Private unary execution unit produced by the Stage Barrier compiler."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=STAGE_PATH_NODE_ID,
            display_name="Stage Path (Internal)",
            category="Turing Utils/internal",
            description=(
                "Internal one-input/one-output Stage Barrier route. The server "
                "creates this node while compiling a submitted workflow."
            ),
            is_dev_only=True,
            inputs=[
                io.Int.Input(
                    "stage",
                    default=0,
                    min=0,
                    max=2**31 - 1,
                    step=1,
                    socketless=True,
                ),
                io.AnyType.Input("value", optional=True),
            ],
            outputs=[io.AnyType.Output(display_name="value")],
        )

    @classmethod
    def execute(cls, stage, value=None) -> io.NodeOutput:
        stage = int(stage)
        if stage < 0:
            raise ValueError("Stage Path stage must be greater than or equal to zero")
        LOG.info("Stage Barrier path reached: stage=%d", stage)
        return io.NodeOutput(value)
