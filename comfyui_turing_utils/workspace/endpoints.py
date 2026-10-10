"""Dynamic pass-through endpoints; port identity is independent of display order."""

import json

from .protocol import MATERIAL_TYPES
from .parameters import (
    DYNAMIC_COMBO,
    PARAMETER_TYPES,
    parameter_default,
    validate_definition,
    validate_parameter,
)

DATA_TYPES = {value[0] for value in MATERIAL_TYPES.values()}

INPUTS = "TuringCanvasInputs"
OUTPUTS = "TuringCanvasOutputs"
POSITION = "TURING_CANVAS_POSITION"


class PortInputs(dict):
    def __contains__(self, key):
        return key.startswith("port_") or super().__contains__(key)

    def __getitem__(self, key):
        if key.startswith("port_"):
            return ("*", {"lazy": True})
        return super().__getitem__(key)


class PortTypes(tuple):
    def __getitem__(self, index):
        return "*" if isinstance(index, int) else super().__getitem__(index)


def parse_ports(value, *, endpoint=INPUTS):
    result = json.loads(value)
    if not isinstance(result, list):
        raise ValueError("Endpoint ports must be a list")
    identities, slots = set(), set()
    for port in result:
        if (
            not isinstance(port, dict)
            or not isinstance(port.get("name"), str)
            or not isinstance(port.get("type"), str)
        ):
            raise ValueError("Each endpoint needs a name and ComfyUI type")
        if (
            not isinstance(port.get("id"), str)
            or not port["id"]
            or port["id"] in identities
        ):
            raise ValueError("Endpoint port IDs must be unique")
        slot = port.get("slot")
        if type(slot) is not int or slot < 0 or slot in slots:
            raise ValueError("Endpoint slots must be unique nonnegative integers")
        if port.get("kind") not in {"value", "position", "parameter"}:
            raise ValueError("Unknown endpoint port kind")
        if port["kind"] == "position" and port["type"] != POSITION:
            raise ValueError("Position markers must use the position type")
        if port["kind"] == "value" and port["type"] not in DATA_TYPES:
            raise ValueError(
                "Canvas endpoints only support IMAGE, VIDEO, AUDIO and STRING materials"
            )
        if port["kind"] == "parameter":
            if endpoint != INPUTS or port["type"] not in PARAMETER_TYPES:
                raise ValueError("Only Canvas Inputs supports editable parameters")
            if not isinstance(port.get("options", {}), dict):
                raise ValueError("Parameter options must be an object")
            validate_definition(port)
            validate_parameter(port, parameter_default(port))
        elif "default" in port:
            if port["type"] != "STRING" or not isinstance(port["default"], str):
                raise ValueError("Only text materials may have an inline default")
        identities.add(port["id"])
        slots.add(slot)
    if slots != set(range(len(result))):
        raise ValueError("Endpoint slots must form a contiguous range")
    return result


class CanvasInputs:
    CATEGORY = "Turing Utils/Materials"
    FUNCTION = "forward"
    RETURN_TYPES = PortTypes(("*",))
    ENDPOINT = INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ports": (
                    "STRING",
                    {"default": "[]", "hidden": True, "socketless": True},
                )
            },
            "optional": PortInputs(),
        }

    @classmethod
    def VALIDATE_INPUTS(cls, ports, input_types):
        try:
            entries = parse_ports(ports, endpoint=cls.ENDPOINT)
        except ValueError as error:
            return str(error)
        allowed = {
            f"port_{p['slot']}": p["type"]
            for p in entries
            if p["kind"] != "position" and p["type"] != DYNAMIC_COMBO
        }
        for name, received in input_types.items():
            if name not in allowed or received not in {allowed[name], "*"}:
                return "Connected input does not match its endpoint port"
        return True

    def check_lazy_status(self, ports, **kwargs):
        return [
            f"port_{p['slot']}"
            for p in parse_ports(ports, endpoint=self.ENDPOINT)
            if p["kind"] != "position"
            and f"port_{p['slot']}" in kwargs
            and kwargs[f"port_{p['slot']}"] is None
        ]

    def forward(self, ports, **kwargs):
        entries = parse_ports(ports, endpoint=self.ENDPOINT)
        result = [None] * (max((p["slot"] for p in entries), default=-1) + 1)
        for port in entries:
            if port["kind"] != "position":
                result[port["slot"]] = kwargs.get(
                    f"port_{port['slot']}", parameter_default(port)
                )
        return tuple(result)


class CanvasOutputs(CanvasInputs):
    OUTPUT_NODE = True
    ENDPOINT = OUTPUTS


ENDPOINT_NODES = {INPUTS: CanvasInputs, OUTPUTS: CanvasOutputs}
