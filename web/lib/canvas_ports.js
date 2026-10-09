export const INPUTS = "TuringCanvasInputs";
export const OUTPUTS = "TuringCanvasOutputs";
export const POSITION = "TURING_CANVAS_POSITION";
export function entries(node) {
  return JSON.parse(node.widgets.find((w) => w.name === "ports").value || "[]");
}
export function syncPorts(node, ports) {
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
  node.graph?.setDirtyCanvas(true, true);
}
export function addPort(node, name, type = "STRING", kind = "value") {
  const ports = entries(node),
    port = { id: crypto.randomUUID(), slot: ports.length, name, type, kind };
  ports.push(port);
  syncPorts(node, ports);
  return port;
}
export function editPorts(node) {
  const dialog = document.createElement("dialog"),
    list = document.createElement("div");
  dialog.style.cssText =
    "max-height:80vh;overflow:auto;background:var(--comfy-menu-bg);color:var(--input-text);padding:16px";
  dialog.append(list);
  let dragged;
  const draw = () => {
    list.replaceChildren();
    for (const port of entries(node)) {
      const row = document.createElement("div");
      row.draggable = true;
      row.style.cssText = "display:flex;gap:8px;padding:8px";
      const handle = document.createElement("span");
      handle.textContent = "☰";
      row.append(handle);
      const name = document.createElement("input");
      name.value = port.name;
      name.onchange = () => {
        const ps = entries(node);
        ps.find((p) => p.id === port.id).name = name.value;
        syncPorts(node, ps);
      };
      row.append(name);
      const label = document.createElement("span");
      label.textContent = port.kind === "position" ? "素材桩位置" : port.type;
      row.append(label);
      if (
        port.kind === "value" &&
        ["STRING", "INT", "FLOAT", "BOOLEAN"].includes(port.type)
      ) {
        const value = document.createElement("input");
        value.placeholder = "未连接时的默认值";
        value.value = port.default ?? "";
        value.onchange = () => {
          const ps = entries(node),
            p = ps.find((p) => p.id === port.id);
          p.default =
            p.type === "STRING"
              ? value.value
              : p.type === "BOOLEAN"
                ? value.value === "true"
                : Number(value.value);
          syncPorts(node, ps);
        };
        row.append(value);
      }
      row.ondragstart = (e) => {
        dragged = port.id;
        e.dataTransfer.setData("text/plain", port.id);
      };
      row.ondragover = (e) => e.preventDefault();
      row.ondrop = (e) => {
        e.preventDefault();
        const ps = entries(node),
          from = ps.findIndex((p) => p.id === dragged),
          to = ps.findIndex((p) => p.id === port.id);
        if (from < 0) return;
        ps.splice(to, 0, ps.splice(from, 1)[0]);
        syncPorts(node, ps);
        draw();
      };
      const remove = document.createElement("button");
      remove.textContent = "删除";
      remove.onclick = () => {
        if (!confirm("删除端点会断开对应连线，继续？")) return;
        syncPorts(
          node,
          entries(node).filter((p) => p.id !== port.id),
        );
        draw();
      };
      row.append(remove);
      list.append(row);
    }
  };
  for (const kind of node.type === INPUTS ? ["value", "position"] : ["value"]) {
    const add = document.createElement("button");
    add.textContent = kind === "position" ? "添加桩位置" : "添加参数 / 数据";
    add.onclick = () => {
      const name = prompt("显示名称");
      if (!name) return;
      const type =
        kind === "position"
          ? POSITION
          : prompt(
              "ComfyUI 类型，例如 STRING / INT / FLOAT / IMAGE / AUDIO",
              "STRING",
            );
      if (type) {
        addPort(node, name, type, kind);
        draw();
      }
    };
    dialog.append(add);
  }
  const close = document.createElement("button");
  close.textContent = "完成";
  close.onclick = () => dialog.close();
  dialog.append(close);
  dialog.onclose = () => dialog.remove();
  draw();
  document.body.append(dialog);
  dialog.showModal();
}
