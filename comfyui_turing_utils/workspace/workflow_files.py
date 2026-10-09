"""Independent workflow templates and per-project copies; API graphs are derived snapshots."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import uuid

from .cards import describe, flatten
from .compiler import MATERIALS, is_link
from .endpoints import INPUTS, parse_ports

LOCK = threading.RLock()
KEY = "turing_card"


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(workflow):
    source = copy.deepcopy(workflow)
    source.get("extra", {}).pop(KEY, None)
    if not source.get("extra"):
        source.pop("extra", None)
    return hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def pack(workflow, prompt):
    describe(prompt)
    result = copy.deepcopy(workflow)
    result.setdefault("extra", {})[KEY] = {"version": 2, "prompt": copy.deepcopy(prompt),
        "source_hash": digest(result), "overrides": {}}
    return result


def unpack(workflow):
    data = workflow.get("extra", {}).get(KEY)
    if not data or data.get("version") not in {1, 2}:
        raise ValueError("Open this workflow in ComfyUI and save it as a Canvas card first")
    if data["source_hash"] != digest(workflow):
        raise ValueError("Workflow was edited externally; open it and Save Back to Card to refresh the execution graph")
    if data["version"] == 1 and any(n["class_type"] == "TuringMaterialVideo" for n in data["prompt"].values()):
        raise ValueError("This card uses legacy IMAGE/AUDIO video outputs. Open its workflow, reconnect the VIDEO output using native video component nodes, then save the card again. Material files are unchanged.")
    interface = describe(data["prompt"])
    return data, interface


def card_path(root, identity):
    if not re.fullmatch(r"[a-f0-9]{32}", identity):
        raise ValueError("Invalid card instance ID")
    root = Path(root).resolve()
    path = (root / "cards" / identity / "workflow.json").resolve()
    if not path.is_relative_to(root):
        raise ValueError("Card file must stay inside its project")
    return path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def add_instance(root, workflow, title):
    unpack(workflow)
    with LOCK:
        path = Path(root) / "canvas.json"
        canvas = load_json(path) if path.exists() else {"format": 2, "revision": 0, "cards": []}
        identity = uuid.uuid4().hex
        atomic_json(card_path(root, identity), workflow)
        canvas["cards"].append({"id": identity, "title": title, "x": len(canvas["cards"])*420, "y": 0})
        canvas["revision"] += 1
        atomic_json(path, canvas)
        return identity


def read_instance(root, identity):
    path = card_path(root, identity)
    workflow = load_json(path)
    # Opening an externally edited file is allowed; only execution rejects stale snapshots.
    return {"workflow": workflow, "revision": hashlib.sha256(path.read_bytes()).hexdigest()}


def save_instance(root, identity, workflow, prompt, revision):
    with LOCK:
        current = read_instance(root, identity)
        if current["revision"] != revision:
            raise ValueError("Card file changed; reopen it before saving")
        result = pack(workflow, prompt)
        canvas = load_json(Path(root) / "canvas.json")
        _, interface = unpack(result)
        previous = current["workflow"]["extra"][KEY]
        # Read the old identities without applying the new output schema: a
        # legacy IMAGE/AUDIO video card must remain repairable in the editor.
        old_prompt = previous["prompt"]
        old_interface = {
            "stubs": {n["inputs"]["stub_id"]: key for key, n in old_prompt.items()
                      if n["class_type"] in MATERIALS},
            "inputs": next(parse_ports(n["inputs"]["ports"]) for n in old_prompt.values()
                           if n["class_type"] == INPUTS),
        }
        for stub in old_interface["stubs"].keys() & interface["stubs"].keys():
            old = previous["prompt"][old_interface["stubs"][stub]]["class_type"]
            if old != prompt[interface["stubs"][stub]]["class_type"]:
                raise ValueError("A material's stable ID cannot change media type")
        exports = {tuple(prompt[interface["output_node"]]["inputs"][f"port_{p['slot']}"]) for p in interface["outputs"]}
        for connection in canvas.get("connections", []):
            if connection["target"] == identity:
                port = next((p for p in interface["inputs"] if p["id"] == connection["port"] and p["kind"] == "value"), None)
                old = next(p for p in old_interface["inputs"] if p["id"] == connection["port"])
                if port is None or port["type"] != old["type"]:
                    raise ValueError("Disconnect the card input before deleting it or changing its type")
            if connection["source"].startswith(identity + ":"):
                stub = connection["source"].split(":", 1)[1]
                if previous.get("version") == 1 and old_prompt[old_interface["stubs"][stub]]["class_type"] == "TuringMaterialVideo":
                    raise ValueError("Disconnect legacy video outputs before changing them to VIDEO")
                if (interface["stubs"].get(stub), connection.get("slot", 0)) not in exports:
                    raise ValueError("Disconnect the card output before removing its material export")
        atomic_json(card_path(root, identity), result)
        canvas["revision"] += 1
        atomic_json(Path(root) / "canvas.json", canvas)
        return read_instance(root, identity)["revision"]


def document(root):
    canvas = load_json(Path(root) / "canvas.json")
    if canvas.get("format") != 2:
        raise ValueError("Unsupported Canvas project format")
    prompt, cards = {}, []
    for instance in canvas["cards"]:
        try:
            workflow = load_json(card_path(root, instance["id"]))
            data, interface = unpack(workflow)
        except (ValueError, OSError) as error:
            cards.append({**instance, "materials":[], "fields":[], "layout":[], "ports":[], "outputs":[],
                          "instance":True, "error":str(error)})
            continue
        local = flatten(data["prompt"], interface, data.get("overrides"))
        identities = {node: f"{instance['id']}:{stub}" for stub, node in interface["stubs"].items()}
        identities.update({node:f"{instance['id']}:node:{node}" for node in local if node not in identities})
        for node_id, node in local.items():
            for value in node["inputs"].values():
                if is_link(value):
                    value[0] = identities[value[0]]
            prompt[identities[node_id]] = node
        # External bindings replace endpoint defaults without running their local branches.
        bindings = {c["port"]: c for c in canvas.get("connections", []) if c["target"] == instance["id"]}
        if bindings:
            replacements = {p["id"]:[bindings[p["id"]]["source"],bindings[p["id"]].get("slot",0)] for p in interface["inputs"] if p["id"] in bindings}
            # Apply links after local ID namespacing so cross-card material IDs stay intact.
            for node_id,node in data["prompt"].items():
                if node_id not in local:continue
                for name,value in node["inputs"].items():
                    if is_link(value) and value[0] == interface["input_node"]:
                        port = next(p for p in interface["inputs"] if p["slot"] == value[1])
                        if port["id"] in replacements:
                            prompt[identities[node_id]]["inputs"][name] = replacements[port["id"]]
        materials, fields = [], []
        configuration_id = f"{instance['id']}:parameters"
        parameters = {}
        layout = []
        for port in interface["inputs"]:
            if port["kind"] == "position":
                material = identities[interface["stubs"][interface["positions"][port["id"]]]]
                materials.append(material)
                layout.append({"material": material})
            elif port["type"] in {"STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"}:
                value = data.get("overrides", {}).get(port["id"], port.get("default"))
                if value is None:
                    value = {"STRING":"", "INT":0, "FLOAT":0., "BOOLEAN":False, "COMBO":""}[port["type"]]
                parameters[port["id"]] = value
                field = {"node": configuration_id, "input": port["id"], "label": port["name"], "type":port["type"], "options":port.get("options"),
                         "connected": port["id"] in bindings or f"port_{port['slot']}" in data["prompt"][interface["input_node"]]["inputs"] and port["id"] not in data.get("overrides", {})}
                fields.append(field)
                layout.append({"field": field})
        prompt[configuration_id] = {"class_type":"CanvasParameters", "inputs":parameters}
        outputs = []
        for port in interface["outputs"]:
            source, slot = data["prompt"][interface["output_node"]]["inputs"][f"port_{port['slot']}"]
            outputs.append({**port, "source":identities[source], "source_slot":slot})
        cards.append({**instance, "materials": materials, "fields": fields, "layout":layout, "instance":True, "outputs":outputs,
            "ports":[p for p in interface["inputs"] if p["kind"] == "value"], "bindings":bindings})
    return {"revision":canvas["revision"], "prompt":prompt, "workflow":{}, "cards":cards, "format":2}


def connect(root, revision, target, port_id, source, source_slot=None):
    with LOCK:
        path=Path(root)/"canvas.json"
        canvas=load_json(path)
        if canvas["revision"] != revision:
            raise ValueError("Canvas changed; reload before connecting")
        workflow=load_json(card_path(root,target))
        _,interface=unpack(workflow)
        port=next((p for p in interface["inputs"] if p["id"] == port_id and p["kind"] == "value"),None)
        if port is None:
            raise ValueError("Unknown input port")
        connections=[c for c in canvas.get("connections",[]) if (c["target"],c["port"]) != (target,port_id)]
        if source:
            doc=document(root)
            graph=doc["prompt"]
            kind=graph.get(source,{}).get("class_type")
            allowed={"TuringMaterialText":{"STRING"},"TuringMaterialImage":{"IMAGE"},
                     "TuringMaterialVideo":{"VIDEO"},"TuringMaterialAudio":{"AUDIO"}}
            if port["type"] not in allowed.get(kind,set()):
                raise ValueError("Material type does not match this input")
            slot=0
            if source_slot is not None and source_slot != slot:
                raise ValueError("Selected output modality does not match the input")
            if not any(p["source"] == source and p["source_slot"] == slot for c in doc["cards"] for p in c["outputs"]):
                raise ValueError("Source must be exported by Canvas Outputs")
            connections.append({"target":target,"port":port_id,"source":source,"slot":slot})
        canvas["connections"]=connections
        canvas["revision"]+=1
        atomic_json(path,canvas)
        return canvas["revision"]


def patch(root, revision, changes):
    with LOCK:
        path = Path(root) / "canvas.json"
        canvas = load_json(path)
        if canvas["revision"] != revision:
            raise ValueError("Canvas changed; reload before saving")
        pending = {}
        for change in changes:
            if change["type"] == "position":
                card = next(c for c in canvas["cards"] if c["id"] == change["id"])
                x, y = float(change["x"]), float(change["y"])
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("Card coordinates must be finite")
                card.update(x=x, y=y)
            elif change["type"] == "input":
                identity, suffix = change["id"].split(":",1)
                if suffix != "parameters":
                    raise ValueError("Only endpoint parameters can be edited on a card")
                file = card_path(root, identity)
                workflow = pending.setdefault(file, load_json(file))
                data, interface = unpack(workflow)
                port = next((p for p in interface["inputs"] if p["id"] == change["name"] and p["kind"] == "value"), None)
                if port is None:
                    raise ValueError("Unknown card input")
                value = change["value"]
                valid = {"STRING":type(value) is str, "BOOLEAN":type(value) is bool,
                         "INT":type(value) is int, "FLOAT":type(value) in (int,float) and math.isfinite(value),
                         "COMBO": value in port.get("options", [])}
                if not valid.get(port["type"], False):
                    raise ValueError("Value does not match the endpoint parameter type")
                data["overrides"][change["name"]] = change["value"]
            else:
                raise ValueError("Unknown Canvas edit")
        for file, workflow in pending.items():
            atomic_json(file, workflow)
        canvas["revision"] += 1
        atomic_json(path, canvas)
        return canvas["revision"]
