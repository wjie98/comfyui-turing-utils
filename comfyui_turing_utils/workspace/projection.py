"""Request-local card descriptors; computation graphs are expanded only for execution."""

import copy
from pathlib import Path

from .cards import flatten
from .compiler import is_link
from .protocol import MATERIAL_TYPES
from .workflow_files import card_path, load_json, unpack


def document(root, canvas=None, card_ids=None, *, execution=False):
    if canvas is None:
        canvas = load_json(Path(root) / "canvas.json")
    if canvas.get("format") != 3:
        raise ValueError("Unsupported Canvas project format")
    prompt, cards, connections = {}, [], {}
    for connection in canvas.get("connections", []):
        connections.setdefault(connection["target"], {})[connection["port"]] = (
            connection
        )
    for instance in canvas["cards"]:
        if card_ids is not None and instance["id"] not in card_ids:
            continue
        try:
            workflow = load_json(card_path(root, instance["id"]))
            data, interface = unpack(workflow)
        except (ValueError, OSError) as error:
            cards.append(
                {
                    **instance,
                    "materials": [],
                    "fields": [],
                    "layout": [],
                    "ports": [],
                    "outputs": [],
                    "instance": True,
                    "error": str(error),
                }
            )
            continue
        local = (
            flatten(data["prompt"], interface, data.get("overrides"))
            if execution
            else {
                node: copy.deepcopy(data["prompt"][node])
                for node in interface["stubs"].values()
            }
        )
        identities = {
            node: f"{instance['id']}:{stub}"
            for stub, node in interface["stubs"].items()
        }
        identities.update(
            {
                node: f"{instance['id']}:node:{node}"
                for node in data["prompt"]
                if node not in identities
            }
        )
        for node_id, node in local.items():
            for value in node["inputs"].values():
                if is_link(value):
                    value[0] = identities[value[0]]
            prompt[identities[node_id]] = node
        # External bindings replace endpoint defaults without running their local branches.
        bindings = connections.get(instance["id"], {})
        if bindings and execution:
            replacements = {
                p["id"]: [bindings[p["id"]]["source"], bindings[p["id"]].get("slot", 0)]
                for p in interface["inputs"]
                if p["id"] in bindings
            }
            # Apply links after local ID namespacing so cross-card material IDs stay intact.
            for node_id, node in data["prompt"].items():
                if node_id not in local:
                    continue
                for name, value in node["inputs"].items():
                    if is_link(value) and value[0] == interface["input_node"]:
                        port = next(
                            p for p in interface["inputs"] if p["slot"] == value[1]
                        )
                        if port["id"] in replacements:
                            replacement = replacements[port["id"]]
                            prompt[identities[node_id]]["inputs"][name] = replacement
                            if replacement[0] not in prompt:
                                kind = next(
                                    kind
                                    for kind, types in MATERIAL_TYPES.items()
                                    if port["type"] in types
                                )
                                prompt[replacement[0]] = {
                                    "class_type": "TuringMaterial" + kind.title(),
                                    "inputs": {},
                                }
        materials, fields = [], []
        configuration_id = f"{instance['id']}:parameters"
        parameters = {}
        layout = []
        for port in interface["inputs"]:
            if port["kind"] == "position":
                material = identities[
                    interface["stubs"][interface["positions"][port["id"]]]
                ]
                materials.append(material)
                layout.append({"material": material})
            elif port["type"] == "STRING":
                value = data.get("overrides", {}).get(port["id"], port.get("default"))
                if value is None:
                    value = ""
                parameters[port["id"]] = value
                field = {
                    "node": configuration_id,
                    "input": port["id"],
                    "label": port["name"],
                    "type": port["type"],
                    "options": port.get("options"),
                    "connected": port["id"] in bindings
                    or f"port_{port['slot']}"
                    in data["prompt"][interface["input_node"]]["inputs"]
                    and port["id"] not in data.get("overrides", {}),
                }
                fields.append(field)
                layout.append({"field": field})
        prompt[configuration_id] = {
            "class_type": "CanvasParameters",
            "inputs": parameters,
        }
        outputs = []
        for port in interface["outputs"]:
            source, slot = data["prompt"][interface["output_node"]]["inputs"][
                f"port_{port['slot']}"
            ]
            outputs.append({**port, "source": identities[source], "source_slot": slot})
        cards.append(
            {
                **instance,
                "materials": materials,
                "fields": fields,
                "layout": layout,
                "instance": True,
                "outputs": outputs,
                "ports": [p for p in interface["inputs"] if p["kind"] == "value"],
                "bindings": bindings,
            }
        )
    return {
        "revision": canvas["revision"],
        "prompt": prompt,
        "workflow": {},
        "cards": cards,
        "format": 3,
    }
