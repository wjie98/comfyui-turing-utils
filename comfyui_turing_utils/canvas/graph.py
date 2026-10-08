"""Canvas links carry published material, never implied recomputation."""

import hashlib
import json
import re


PREFIX = "TuringCanvas"
SETTINGS = PREFIX + "Settings"
H3_SETTINGS = PREFIX + "H3Settings"
H3 = PREFIX + "H3"
ASSETS = {PREFIX + "Image": "image", PREFIX + "Video": "video", PREFIX + "Audio": "audio"}
PUBLIC = {SETTINGS, H3_SETTINGS, H3, *ASSETS}
PORT_KINDS = {"first_frame": {"image"}, "last_frame": {"image"}, "target": {"video"}, "prefix": {"video"}}


def port_kinds(port):
    match = re.fullmatch(r"(image|video|audio)s\.\1_\d+", port)
    return ({match[1]} if match[1] != "audio" else {"audio", "video"}) if match else PORT_KINDS.get(port, set())


def reference_ports(materials, kind):
    return sorted(
        (p for p in materials if p.startswith(f"{kind}s.{kind}_")), key=lambda p: int(p.rsplit("_", 1)[1]))


def image_ports(materials):
    return reference_ports(materials, "image")


def validate_canvas(graph):
    nodes = graph.get("nodes", [])
    if not nodes:
        raise ValueError("Canvas is empty")
    by_id = {}
    for node in nodes:
        if node.get("type") not in PUBLIC:
            raise ValueError(f"Material canvas cannot contain ordinary nodes: {node.get('type')}")
        key = str(node["id"])
        if key in by_id:
            raise ValueError("Duplicate canvas node ID")
        by_id[key] = node
        if node.get("mode", 0) != 0:
            raise ValueError("Canvas nodes do not support mute/bypass; disconnect the material instead")
    settings = [n for n in nodes if n["type"] == SETTINGS]
    if len(settings) != 1:
        raise ValueError("A material canvas requires exactly one Canvas Settings node")
    models = [n for n in nodes if n["type"] == H3_SETTINGS]
    if len(models) > 1:
        raise ValueError("Use at most one global H3 Settings node")
    for node in nodes:
        for port, link in node.get("inputs", {}).items():
            if not port_kinds(port) or str(link["node"]) not in by_id:
                raise ValueError("Invalid canvas material connection")
            source = by_id[str(link["node"])]
            if source["type"] in {SETTINGS, H3_SETTINGS}:
                raise ValueError("Settings cannot be connected as material")
            slot = int(link.get("slot", 0))
            kind = ASSETS.get(source["type"], "video")
            if slot == 1 and kind == "video":
                kind = "audio"
            elif slot != 0:
                raise ValueError("Invalid material output slot")
            if kind not in port_kinds(port):
                raise ValueError(f"{port} cannot consume {kind} material")
    return by_id, {**(models[0]["values"] if models else {}), **settings[0]["values"]}


def material_inputs(node, nodes, state, project, allow_missing=False):
    result = {}
    for port, link in sorted(node.get("inputs", {}).items()):
        source_id = str(link["node"])
        source = nodes[source_id]
        values = source.get("values", {})
        asset = link.get("asset") or (
            values.get("asset_id") if source["type"] in ASSETS else
            state["tasks"].get(source_id, {}).get("asset"))
        if not asset:
            if allow_missing:
                result[port] = {"missing": source_id}
                continue
            raise ValueError(f"Input {port}: node {source_id} has no published material")
        try:
            data = project.asset(asset)
        except (ValueError, OSError) as error:
            if not allow_missing:
                raise
            result[port] = {"missing": source_id, "error": str(error)}
            continue
        result[port] = {"asset": asset, "slot": int(link.get("slot", 0)),
                        "start": float(values.get("start_seconds", 0)) if source["type"] in ASSETS else 0,
                        "duration": float(values.get("duration_seconds", 0)) if source["type"] in ASSETS else 0}
        if source["type"] in ASSETS:
            result[port].update({key: values[key] for key in ("force_rate", "custom_width", "custom_height",
                "skip_first_frames", "frame_load_cap", "select_every_nth") if key in values})
        if data["kind"] not in port_kinds(port):
            raise ValueError(f"Wrong asset kind for {port}")
    return result


def signature(materials):
    return hashlib.sha256(json.dumps(materials, sort_keys=True).encode()).hexdigest()


def plan(graph, project):
    nodes, _ = validate_canvas(graph)
    state = project.state()
    order, visiting, visited = [], set(), set()

    def visit(key):
        if key in visiting:
            raise ValueError("Canvas refresh requires an acyclic material graph")
        if key in visited:
            return
        visiting.add(key)
        for link in nodes[key].get("inputs", {}).values():
            if not link.get("asset"):
                visit(str(link["node"]))
        visiting.remove(key)
        visited.add(key)
        order.append(key)

    for key in nodes:
        visit(key)
    scheduled, result = set(), []
    for key in order:
        node = nodes[key]
        if node["type"] != H3:
            continue
        materials = material_inputs(node, nodes, state, project, allow_missing=True)
        blocked = any("missing" in m and m["missing"] not in scheduled for m in materials.values())
        previous = state["tasks"].get(key)
        upstream = any(str(l["node"]) in scheduled and not l.get("asset") for l in node.get("inputs", {}).values())
        dirty = not previous or previous["signature"] != signature(materials)
        status = "blocked" if blocked else "changed" if dirty else "upstream" if upstream else "current"
        if status in {"changed", "upstream"}:
            scheduled.add(key)
        result.append({"id": key, "status": status})
    return result
