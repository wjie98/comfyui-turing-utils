"""Validate workflow interfaces and flatten endpoint wiring for task submission."""

import copy

from .compiler import MATERIALS, is_link
from .endpoints import INPUTS, OUTPUTS, parse_ports


def describe(prompt):
    inputs = [key for key, node in prompt.items() if node["class_type"] == INPUTS]
    outputs = [key for key, node in prompt.items() if node["class_type"] == OUTPUTS]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("A card needs exactly one Canvas Inputs and one Canvas Outputs")
    input_id, output_id = inputs[0], outputs[0]
    incoming = parse_ports(prompt[input_id]["inputs"]["ports"])
    outgoing = parse_ports(prompt[output_id]["inputs"]["ports"])
    stubs, positioned, exported = {}, {}, set()
    for key, node in prompt.items():
        if node["class_type"] not in MATERIALS:
            continue
        identity = node["inputs"].get("stub_id", "")
        if not identity or identity in stubs:
            raise ValueError("Every material needs a unique stable stub_id")
        stubs[identity] = key
        marker = node["inputs"].get("position")
        if not is_link(marker) or marker[0] != input_id:
            raise ValueError(f"Material {identity} has no Canvas Inputs position binding")
        port = next((p for p in incoming if p["slot"] == marker[1]), None)
        if port is None or port["kind"] != "position" or port["id"] in positioned:
            raise ValueError("Each position port must bind exactly one material")
        positioned[port["id"]] = identity
    for port in incoming:
        if port["kind"] == "position" and port["id"] not in positioned:
            raise ValueError("Unbound material position")
    for port in outgoing:
        source = prompt[output_id]["inputs"].get(f"port_{port['slot']}")
        if port["kind"] != "value" or not is_link(source) or source[0] not in stubs.values():
            raise ValueError("Canvas Outputs only accepts material outputs")
        node = prompt[source[0]]
        count = 2 if node["class_type"] == "TuringMaterialVideo" else 1
        if not 0 <= source[1] < count:
            raise ValueError("Invalid material output slot")
        types = {"text":("STRING",), "image":("IMAGE",), "video":("IMAGE","AUDIO"), "audio":("AUDIO",)}
        if port["type"] != types[MATERIALS[node["class_type"]]][source[1]]:
            raise ValueError("Exported port type does not match its material output")
        exported.add(source[0])
    if exported != set(stubs.values()):
        raise ValueError("Every material must be connected to Canvas Outputs")
    return {"input_node": input_id, "output_node": output_id, "inputs": incoming,
            "outputs": outgoing, "stubs": stubs, "positions": positioned}


def flatten(prompt, interface, overrides=None):
    result = copy.deepcopy(prompt)
    inputs = result[interface["input_node"]]["inputs"]
    replacements = {}
    for port in interface["inputs"]:
        if port["kind"] != "value":
            continue
        value = (overrides or {}).get(port["id"], inputs.get(f"port_{port['slot']}", port.get("default")))
        replacements[port["slot"]] = value
    for node in result.values():
        if node["class_type"] in MATERIALS:
            node["inputs"].pop("position", None)
        for name, value in list(node["inputs"].items()):
            if is_link(value) and value[0] == interface["input_node"]:
                if value[1] not in replacements:
                    raise ValueError("A position marker cannot be used as computation data")
                replacement = replacements[value[1]]
                if replacement is None:
                    del node["inputs"][name]
                else:
                    node["inputs"][name] = copy.deepcopy(replacement)
    del result[interface["input_node"]]
    del result[interface["output_node"]]
    return result
