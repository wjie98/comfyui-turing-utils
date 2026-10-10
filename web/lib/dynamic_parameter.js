import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

export const DYNAMIC_COMBO = "COMFY_DYNAMICCOMBO_V3";
const SCALARS = new Set(["INT", "FLOAT", "BOOLEAN", "STRING", "COMBO"]);

function fields(option) {
  return Object.entries({ ...option.inputs?.required, ...option.inputs?.optional });
}
function scalarSpec(spec) {
  const [type, options = {}] = spec;
  return Array.isArray(type)
    ? ["COMBO", { ...options, values: type }]
    : [type, type === "COMBO" ? { ...options, values: options.options } : options];
}
function defaultValue(spec) {
  const [type, options] = scalarSpec(spec);
  if (type === DYNAMIC_COMBO)
    return {
      selected: options.options[0].key,
      branches: Object.fromEntries(
        options.options.map((o) => [
          o.key,
          Object.fromEntries(
            fields(o).map(([name, spec]) => [name, defaultValue(spec)]),
          ),
        ]),
      ),
    };
  return (
    options.default ??
    (type === "COMBO"
      ? (options.values[0] ?? null)
      : { INT: 0, FLOAT: 0, BOOLEAN: false, STRING: "" }[type])
  );
}
function editable(spec, depth = 0) {
  if (depth > 16) return false;
  const [type, options] = scalarSpec(spec);
  if (options.forceInput || options.multiselect) return false;
  if (type !== DYNAMIC_COMBO) {
    if (!SCALARS.has(type)) return false;
    return (
      type !== "COMBO" ||
      (Array.isArray(options.values) &&
        options.values.every(
          (v) =>
            ["string", "number", "boolean"].includes(typeof v) &&
            (typeof v !== "number" || Number.isFinite(v)),
        ))
    );
  }
  return (
    Array.isArray(options.options) &&
    options.options.length > 0 &&
    options.options.every((o) => fields(o).every(([, s]) => editable(s, depth + 1)))
  );
}
export function dynamicSpec(node, name) {
  let inputs = node.constructor.nodeData?.input;
  let prefix = "";
  let spec;
  for (const part of name.split(".")) {
    spec = inputs?.required?.[part] || inputs?.optional?.[part];
    if (!spec) return null;
    prefix = prefix ? `${prefix}.${part}` : part;
    const option = spec[1]?.options?.find(
      (o) => o.key === node.widgets.find((w) => w.name === prefix)?.value,
    );
    inputs = option?.inputs;
  }
  return spec?.[0] === DYNAMIC_COMBO ? spec : null;
}
function visit(spec, value, name, fn) {
  fn(name, spec, value);
  const option = spec[1].options.find((o) => o.key === value.selected);
  for (const [key, child] of fields(option)) {
    const path = `${name}.${key}`;
    const v = value.branches[value.selected][key];
    if (child[0] === DYNAMIC_COMBO) visit(child, v, path, fn);
    else fn(path, child, v, value.branches[value.selected], key);
  }
}
export function applyDynamicWidgets(node, name, options, value) {
  visit([DYNAMIC_COMBO, options], value, name, (path, spec, v) => {
    const widget = node.widgets.find((w) => w.name === path);
    if (!widget) throw Error(`Missing dynamic parameter: ${path}`);
    const next = spec[0] === DYNAMIC_COMBO ? v.selected : v;
    if (widget.value !== next) widget.value = next;
  });
}
export function dynamicDefinition(node, widget) {
  const spec = dynamicSpec(node, widget.name);
  if (!spec || !editable(spec)) return null;
  const value = defaultValue(spec);
  visit(spec, value, widget.name, (path, child, v, branch, key) => {
    const current = node.widgets.find((w) => w.name === path)?.value;
    if (current === undefined) return;
    if (child[0] === DYNAMIC_COMBO) v.selected = current;
    else branch[key] = current;
  });
  return { type: DYNAMIC_COMBO, kind: "parameter", options: spec[1], default: value };
}

// The native constructor owns branch creation, layout and classic/Vue rendering.
export function dynamicParameterWidget(node, name, value, callback, options) {
  const root = ComfyWidgets[DYNAMIC_COMBO](
    node,
    name,
    [DYNAMIC_COMBO, options],
    app,
  ).widget;
  let state;
  const spec = [DYNAMIC_COMBO, options];
  const owned = (w) => w.name === name || w.name.startsWith(name + ".");
  function sync() {
    applyDynamicWidgets(node, name, options, state);
    // A card form has no computational sockets. Only its material ports are wired.
    for (let i = node.inputs.length - 1; i >= 0; i--)
      if (owned(node.inputs[i])) node.removeInput(i);
    visit(spec, state, name, (path, child, v, branch, key) => {
      const widget = node.widgets.find((w) => w.name === path);
      widget.options = { ...widget.options, serialize: false };
      widget.serialize = false;
      widget._portId = root._portId;
      if (widget._dynamicParameterRoot === root) return;
      widget._dynamicParameterRoot = root;
      if (!("_nativeParameterCallback" in widget))
        widget._nativeParameterCallback = widget.callback;
      const nativeCallback = widget._nativeParameterCallback;
      widget.callback = function (next, ...args) {
        nativeCallback?.call(this, next, ...args);
        if (child[0] === DYNAMIC_COMBO) v.selected = next;
        else branch[key] = next;
        sync();
        node.setSize([node.size[0], node.computeSize()[1]]);
        node.graph?.setDirtyCanvas(true, true);
        return callback(JSON.parse(JSON.stringify(state)));
      };
    });
  }
  root.setParameterValue = (value) => {
    state = JSON.parse(JSON.stringify(value));
    // Branch callbacks capture the current state, so rebind them after restoration.
    for (const w of node.widgets.filter(owned)) delete w._dynamicParameterRoot;
    sync();
  };
  root.setParameterValue(value);
  return root;
}

export function expandDynamicParameters(prompt) {
  const endpoints = new Map(
    Object.entries(prompt)
      .filter(([, node]) => node.class_type === "TuringCanvasInputs")
      .map(([id, node]) => [
        id,
        new Map(JSON.parse(node.inputs.ports).map((p) => [p.slot, p])),
      ]),
  );
  if (!endpoints.size) return;
  for (const node of Object.values(prompt)) {
    const original = { ...node.inputs };
    for (const [name, link] of Object.entries(original)) {
      if (!Array.isArray(link) || link.length !== 2) continue;
      const port = endpoints.get(link[0])?.get(link[1]);
      if (port?.type !== DYNAMIC_COMBO) continue;
      if (
        Object.entries(original).some(
          ([key, v]) => key.startsWith(name + ".") && Array.isArray(v),
        )
      )
        throw Error("Disconnect branch inputs before exposing the whole DynamicCombo");
      for (const key of Object.keys(node.inputs))
        if (key === name || key.startsWith(name + ".")) delete node.inputs[key];
      visit([DYNAMIC_COMBO, port.options], port.default, name, (path, spec, value) => {
        node.inputs[path] = spec[0] === DYNAMIC_COMBO ? value.selected : value;
      });
      node._meta ??= {};
      node._meta.turing_dynamic_inputs ??= {};
      node._meta.turing_dynamic_inputs[name] = link;
    }
  }
}
