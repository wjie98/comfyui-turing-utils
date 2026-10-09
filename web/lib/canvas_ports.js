export const INPUTS = "TuringCanvasInputs";
export const OUTPUTS = "TuringCanvasOutputs";
export const POSITION = "TURING_CANVAS_POSITION";
export function entries(node) {
  return JSON.parse(node.widgets.find((w) => w.name === "ports").value || "[]");
}
export function syncPorts(node, ports) {
  node._syncingPorts = true;
  for (let i = (node.inputs?.length || 0) - 1; i >= 0; i--)
    if (node.inputs[i]._append || node.inputs[i].name === "＋")
      node.removeInput(i);
  for (let i = (node.outputs?.length || 0) - 1; i >= 0; i--)
    if (node.outputs[i]._append || node.outputs[i].name === "＋")
      node.removeOutput(i);
  const old = entries(node),
    ids = new Map(old.map((p) => [p.slot, p.id]));
  for (const input of node.inputs || [])
    if (input.name.startsWith("port_"))
      input._portId ??= ids.get(Number(input.name.slice(5)));
  (node.outputs || []).forEach((output, i) => (output._portId ??= ids.get(i)));
  const wanted = new Set(ports.map((p) => p.id));
  for (let i = (node.inputs?.length || 0) - 1; i >= 0; i--) {
    const s = node.inputs[i];
    if (
      s.name.startsWith("port_") &&
      (!wanted.has(s._portId) ||
        ports.find((p) => p.id === s._portId)?.kind === "position")
    )
      node.removeInput(i);
  }
  for (let i = (node.outputs?.length || 0) - 1; i >= 0; i--)
    if (!wanted.has(node.outputs[i]._portId)) node.removeOutput(i);
  for (const p of ports) {
    if (p.kind === "value" && !node.inputs.some((s) => s._portId === p.id)) {
      node.addInput("port_" + p.slot, p.type);
      node.inputs.at(-1)._portId = p.id;
    }
    if (!node.outputs.some((s) => s._portId === p.id)) {
      node.addOutput(p.name, p.kind === "position" ? POSITION : p.type);
      node.outputs.at(-1)._portId = p.id;
    }
  }
  const order = new Map(ports.map((p, i) => [p.id, i]));
  // Modern LiteGraph derives connectivity from slot indices. Snapshot links
  // before moving descriptors; reading slot.links afterwards reads another port.
  const inputLinks = new Map(
    node.inputs.map((s) => [s._portId, node.graph?.links[s.link]]),
  );
  const outputLinks = new Map(
    node.outputs.map((s) => [
      s._portId,
      (s.links || []).map((id) => node.graph?.links[id]).filter(Boolean),
    ]),
  );
  // Vacate target slots before assigning their new indices (input slots are unique).
  for (const [id, link] of inputLinks)
    if (link) link.target_slot = node.inputs.length + order.get(id);
  node.inputs.sort(
    (a, b) => (order.get(a._portId) ?? -1) - (order.get(b._portId) ?? -1),
  );
  node.outputs.sort((a, b) => order.get(a._portId) - order.get(b._portId));
  ports.forEach((p, i) => (p.slot = i));
  for (let i = 0; i < node.inputs.length; i++) {
    const s = node.inputs[i],
      p = ports.find((p) => p.id === s._portId);
    if (!p) continue;
    s.name = "port_" + p.slot;
    s.label = p.name;
    s.type = p.type;
    const link = inputLinks.get(s._portId);
    if (link) link.target_slot = i;
  }
  for (let i = 0; i < node.outputs.length; i++) {
    const s = node.outputs[i],
      p = ports[i];
    s.name = p.name;
    s.type = p.kind === "position" ? POSITION : p.type;
    for (const link of outputLinks.get(s._portId) || []) link.origin_slot = i;
  }
  node.widgets.find((w) => w.name === "ports").value = JSON.stringify(ports);
  node.setSize(node.computeSize());
  node._syncingPorts = false;
  appendSocket(node);
  node.graph?.setDirtyCanvas(true, true);
}
export function appendSocket(node) {
  if (node._appendingSocket || node._syncingPorts) return;
  node._appendingSocket = true;
  const inputs = node.comfyClass === OUTPUTS || node.type === OUTPUTS;
  const slots = inputs ? node.inputs : node.outputs;
  if (!slots?.some((s) => s._append)) {
    if (inputs) node.addInput("＋", "*");
    else node.addOutput("＋", "*");
    (inputs ? node.inputs : node.outputs).at(-1)._append = true;
  }
  for (const s of [...node.inputs, ...node.outputs]) delete s.pos;
  node._appendingSocket = false;
}
export function addPort(node, name, type = "STRING", kind = "value") {
  const ports = entries(node),
    port = { id: crypto.randomUUID(), slot: ports.length, name, type, kind };
  ports.push(port);
  syncPorts(node, ports);
  return port;
}
