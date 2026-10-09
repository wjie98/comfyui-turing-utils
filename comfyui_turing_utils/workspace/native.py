"""Native editor projection. Project storage remains authoritative for material content."""

import copy
import math

from . import workflow_files as files

PROJECT = "TuringCanvasProject"
CARD = "TuringCanvasCard"


def project_workflow(project):
    with files.LOCK:
        canvas = files.load_json(project.root / "canvas.json")
        document = project.document()
        positions = canvas.get("layout", {})
        nodes = [{"id":1, "type":PROJECT, "pos":[20,20], "size":[310,110],
                  "flags":{"pinned":True}, "properties":{"directory":project.directory}, "order":0, "mode":0}]
        for i, card in enumerate(document["cards"], 2):
            layout = positions.get(card["id"], {})
            nodes.append({"id":i, "type":CARD, "title":card["title"],
                          "pos":layout.get("pos", [360+(i-2)*350, 20]),
                          "size":layout.get("size", [320,300]), "flags":{}, "order":i-1, "mode":0,
                          "properties":{"instance":card["id"]}})
        # Runtime descriptors are re-read on opening; they are not a second editable graph.
        workflow = {"id":canvas["id"], "version":0.4,
                    "last_node_id":len(nodes), "last_link_id":0, "nodes":nodes, "links":[], "groups":canvas.get("groups", []),
                    "extra":{"turing_project":{"directory":project.directory, "revision":canvas["revision"]},
                             "ds":canvas.get("view", {"scale":1, "offset":[0,0]})}}
        return {"workflow":workflow, "document":document}


def save_layout(project, workflow, revision):
    with files.LOCK:
        path = project.root / "canvas.json"
        canvas = files.load_json(path)
        if canvas["revision"] != revision:
            raise ValueError("Project changed; reopen before saving")
        nodes = workflow.get("nodes", [])
        if sum(n["type"] == PROJECT for n in nodes) != 1 or any(n["type"] not in {PROJECT,CARD} for n in nodes):
            raise ValueError("A Canvas project needs one project node and Canvas cards only")
        cards = {c["id"]:c for c in canvas["cards"]}
        layout, node_ids = {}, {}
        for node in nodes:
            if node["type"] != CARD:
                continue
            identity = node["properties"]["instance"]
            if identity not in cards or identity in layout:
                raise ValueError("Unknown or duplicate card instance; use Add Node to create a copy")
            for key in ("pos", "size"):
                values = node[key]
                if len(values) != 2 or any(not isinstance(v,(float,int)) or not math.isfinite(v) for v in values):
                    raise ValueError("Invalid node geometry")
            layout[identity] = {"pos":node["pos"], "size":node["size"]}
            node_ids[node["id"]] = identity
        doc = project.document()
        descriptors = {c["id"]:c for c in doc["cards"]}
        connections, targets = [], set()
        for link in workflow.get("links", []):
            _, source_id, source_slot, target_id, target_slot, _ = link
            source, target = node_ids[source_id], node_ids[target_id]
            if type(source_slot) is not int or type(target_slot) is not int or source_slot < 0 or target_slot < 0:
                raise ValueError("Invalid material slot")
            if (target_id,target_slot) in targets:
                raise ValueError("A card input can have only one source")
            targets.add((target_id,target_slot))
            if source_slot >= len(descriptors[source]["outputs"]) or target_slot >= len(descriptors[target]["ports"]):
                raise ValueError("Unknown material slot")
            out = descriptors[source]["outputs"][source_slot]
            port = descriptors[target]["ports"][target_slot]
            if out["type"] != port["type"]:
                raise ValueError("Connected material types differ")
            connections.append({"source":out["source"], "slot":out["source_slot"], "target":target, "port":port["id"]})
        canvas["cards"] = [c for c in canvas["cards"] if c["id"] in layout]
        canvas["layout"], canvas["connections"] = layout, connections
        canvas["view"] = copy.deepcopy(workflow.get("extra", {}).get("ds", {}))
        canvas["groups"] = copy.deepcopy(workflow.get("groups", []))
        canvas["revision"] += 1
        files.atomic_json(path, canvas)
        return {"revision":canvas["revision"]}
