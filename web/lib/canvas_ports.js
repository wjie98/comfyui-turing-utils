import { parameterWidget } from "./parameter_widget.js";
import {
  DYNAMIC_COMBO,
  dynamicDefinition,
  dynamicSpec,
  applyDynamicWidgets,
} from "./dynamic_parameter.js";

export const INPUTS = "TuringCanvasInputs";
export const OUTPUTS = "TuringCanvasOutputs";
export const POSITION = "TURING_CANVAS_POSITION";
export const MATERIAL_TYPES = new Set(["STRING", "IMAGE", "VIDEO", "AUDIO"]);
export const PARAMETER_TYPES = new Set([
  "INT",
  "FLOAT",
  "BOOLEAN",
  "STRING",
  "COMBO",
  DYNAMIC_COMBO,
]);

export function parameterDefinition(type, widget, target) {
  if (type === DYNAMIC_COMBO)
    return target && widget ? dynamicDefinition(target, widget) : null;
  const values = Array.isArray(type) ? type : widget?.options?.values;
  if (Array.isArray(type)) type = "COMBO";
  if (!PARAMETER_TYPES.has(type)) return null;
  if (
    type === "COMBO" &&
    (!Array.isArray(values) ||
      !values.length ||
      values.some(
        (v) =>
          !["string", "number", "boolean"].includes(typeof v) ||
          (typeof v === "number" && !Number.isFinite(v)),
      ))
  )
    return null;
  const options = {};
  for (const key of ["min", "max", "step", "step2", "precision", "round", "on", "off"])
    if (widget?.options?.[key] !== undefined) options[key] = widget.options[key];
  if (type === "STRING") options.multiline = widget?.inputEl?.tagName === "TEXTAREA";
  if (type === "COMBO") options.values = values;
  const value =
    widget?.value ??
    (type === "COMBO"
      ? values[0]
      : { INT: 0, FLOAT: 0, BOOLEAN: false, STRING: "" }[type]);
  return { type, kind: "parameter", default: value, options };
}

function syncParameterWidgets(node, ports) {
  const wanted = ports.filter((p) => p.kind === "parameter");
  for (const widget of [...node.widgets]) {
    if (!widget._portId || wanted.some((p) => p.id === widget._portId)) continue;
    node.removeWidget(widget);
  }
  for (const port of wanted) {
    let widget = node.widgets.find(
      (w) =>
        w._portId === port.id &&
        (!w._dynamicParameterRoot || w._dynamicParameterRoot === w),
    );
    if (!widget) {
      widget = parameterWidget(
        node,
        `parameter:${port.id}`,
        port.type,
        port.default,
        (value) => {
          const ports = entries(node);
          ports.find((p) => p.id === port.id).default = value;
          node.widgets.find((w) => w.name === "ports").value = JSON.stringify(ports);
          syncDynamicTargets(
            node,
            ports.find((p) => p.id === port.id),
          );
          node.graph?.setDirtyCanvas(true, true);
        },
        port.options,
      );
      widget._portId = port.id;
    }
    widget.label = port.name;
    if (widget.setParameterValue) widget.setParameterValue(port.default);
    else widget.value = port.default;
    for (const w of node.widgets)
      if (w._dynamicParameterRoot === widget) w._portId = port.id;
  }
  const order = new Map(wanted.map((p, i) => [p.id, i]));
  node.widgets.sort(
    (a, b) => (order.get(a._portId) ?? -1) - (order.get(b._portId) ?? -1),
  );
}
export function syncDynamicTargets(node, port) {
  if (port.type !== DYNAMIC_COMBO) return;
  const slot = node.outputs.findIndex((s) => s._portId === port.id);
  for (const id of node.outputs[slot]?.links || []) {
    const link = node.graph.links[id];
    const target = node.graph.getNodeById(link.target_id);
    const input = target.inputs[link.target_slot];
    const spec = dynamicSpec(target, input.name);
    if (!spec || JSON.stringify(spec[1]) !== JSON.stringify(port.options))
      throw Error("DynamicCombo schemas do not match; reconnect the parameter group");
    applyDynamicWidgets(target, input.name, port.options, port.default);
  }
}
export function entries(node) {
  return JSON.parse(node.widgets.find((w) => w.name === "ports").value || "[]");
}
export function syncPorts(node, ports) {
  node._syncingPorts = true;
  for (let i = (node.inputs?.length || 0) - 1; i >= 0; i--)
    if (node.inputs[i]._append || node.inputs[i].name === "＋") node.removeInput(i);
  for (let i = (node.outputs?.length || 0) - 1; i >= 0; i--)
    if (node.outputs[i]._append || node.outputs[i].name === "＋") node.removeOutput(i);
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
        ports.find((p) => p.id === s._portId)?.kind === "position" ||
        ports.find((p) => p.id === s._portId)?.type === DYNAMIC_COMBO)
    )
      node.removeInput(i);
  }
  for (let i = (node.outputs?.length || 0) - 1; i >= 0; i--)
    if (!wanted.has(node.outputs[i]._portId)) node.removeOutput(i);
  for (const p of ports) {
    if (
      p.kind !== "position" &&
      p.type !== DYNAMIC_COMBO &&
      !node.inputs.some((s) => s._portId === p.id)
    ) {
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
    s.label = p.name;
    s.type = p.kind === "position" ? POSITION : p.type;
    for (const link of outputLinks.get(s._portId) || []) link.origin_slot = i;
  }
  node.widgets.find((w) => w.name === "ports").value = JSON.stringify(ports);
  syncParameterWidgets(node, ports);
  node._syncingPorts = false;
  for (const port of ports) syncDynamicTargets(node, port);
  appendSocket(node);
  alignInputs(node);
  node.setSize(node.computeSize());
  node.graph?.setDirtyCanvas(true, true);
}
export function appendSocket(node) {
  if (node._appendingSocket || node._syncingPorts) return;
  node._appendingSocket = true;
  for (const inputs of [true, false]) {
    const slots = inputs ? node.inputs : node.outputs;
    if (!slots?.some((s) => s._append)) {
      if (inputs) node.addInput("＋", "*");
      else node.addOutput("＋", "*");
      (inputs ? node.inputs : node.outputs).at(-1)._append = true;
    }
  }
  node._appendingSocket = false;
}
export function alignInputs(node) {
  for (const s of [...node.inputs, ...node.outputs]) delete s.pos;
  for (const input of node.inputs) {
    const index = node.outputs.findIndex((s) =>
      input._append ? s._append : s._portId === input._portId,
    );
    if (index >= 0)
      input.pos = [0, node.getConnectionPos(false, index)[1] - node.pos[1]];
  }
}
export function addPort(node, name, type = "STRING", kind = "value", definition = {}) {
  if (
    (kind === "value" && !MATERIAL_TYPES.has(type)) ||
    (kind === "position" && type !== POSITION) ||
    (kind === "parameter" && (node.type !== INPUTS || !PARAMETER_TYPES.has(type)))
  )
    throw Error("Canvas endpoints only support Image, Video, Audio and Text");
  const ports = entries(node),
    port = {
      ...definition,
      id: crypto.randomUUID(),
      slot: ports.length,
      name,
      type,
      kind,
    };
  ports.push(port);
  syncPorts(node, ports);
  return port;
}
