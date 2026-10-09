"""Ordinary starter workflows, with no generation-model assumptions."""
import json
import uuid

from .endpoints import INPUTS, OUTPUTS, POSITION
from .protocol import MATERIAL_TYPES


def material_template(kinds=("image", "video", "audio", "text")):
    nodes, links, prompt, incoming, outgoing = [], [], {}, [], []
    for i, kind in enumerate(kinds):
        node_id, type_ = i + 3, MATERIAL_TYPES[kind][0]
        incoming.append(dict(id=uuid.uuid4().hex, slot=i, name=kind.title(), kind="position", type=POSITION))
        outgoing.append(dict(id=uuid.uuid4().hex, slot=i, name=kind.title(), kind="value", type=type_))
        values = {"stub_id":uuid.uuid4().hex, **({"text":""} if kind == "text" else {"audio" if kind == "audio" else "file":""})}
        if kind == "video":
            values.update(fps=30., bit_depth="auto", color_space="sRGB", codec="none")
        sockets = ([dict(name="images", type="IMAGE", link=None), dict(name="audio", type="AUDIO", link=None)]
                   if kind == "video" else [dict(name="text" if kind == "text" else "value", type=type_, link=None)])
        sockets.append(dict(name="position", type=POSITION, link=i*2+1))
        nodes.append(dict(id=node_id, type="TuringMaterial"+kind.title(), title=kind.title(), pos=[380,i*320],
            size=[300,260], flags={}, order=i+1, mode=0, properties={}, widgets_values=list(values.values()),
            inputs=sockets, outputs=[dict(name=type_, type=type_, links=[i*2+2])]))
        links.extend([[i*2+1,1,i,node_id,len(sockets)-1,POSITION], [i*2+2,node_id,0,2,i,type_]])
        prompt[str(node_id)] = dict(class_type="TuringMaterial"+kind.title(), inputs={**values,"position":["1",i]}, _meta={"title":kind.title()})
    for node_id, type_, ports, x in [(1,INPUTS,incoming,0), (2,OUTPUTS,outgoing,820)]:
        text = json.dumps(ports)
        nodes.append(dict(id=node_id,type=type_,pos=[x,0],size=[260,220],flags={},order=0,mode=0,properties={},widgets_values=[text],
            inputs=[] if node_id==1 else [dict(name=f"port_{i}",type=p["type"],link=i*2+2) for i,p in enumerate(ports)],
            outputs=[dict(name=p["name"],type=p["type"],links=[i*2+1] if node_id==1 else []) for i,p in enumerate(ports)]))
        prompt[str(node_id)] = dict(class_type=type_,inputs={"ports":text})
        if node_id==2:
            prompt["2"]["inputs"].update({f"port_{i}":[str(i+3),0] for i in range(len(kinds))})
    return dict(version=0.4,last_node_id=len(kinds)+2,last_link_id=len(links),nodes=nodes,links=links,groups=[],config={},extra={}), prompt
