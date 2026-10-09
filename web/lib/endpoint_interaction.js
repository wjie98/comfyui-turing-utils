import {
  INPUTS,
  OUTPUTS,
  POSITION,
  entries,
  syncPorts,
  appendSocket,
} from "./canvas_ports.js";

// All mutation is local to endpoint nodes; the native graph still owns links and undo.
export function installEndpoint(node) {
  const isOutput = (node.comfyClass || node.type) === OUTPUTS;
  const commit = (fn) => {
    node.graph?.beforeChange();
    fn();
    node.graph?.afterChange();
  };
  const grow = (type, name, widget) => {
    const ps = entries(node);
    const p = {
      id: crypto.randomUUID(),
      slot: ps.length,
      name: name || type,
      type: Array.isArray(type) ? "COMBO" : type,
      kind: type === POSITION ? "position" : "value",
    };
    if (widget) {
      p.default = widget.value;
      if (p.type === "COMBO") {
        const values = widget.options?.values;
        p.options = typeof values === "function" ? values() : values;
      }
    }
    if (Array.isArray(type)) p.options = type;
    ps.push(p);
    syncPorts(node, ps);
  };
  node.onConnectInput = function (slot, type, output, source) {
    if (this.inputs[slot]?._append) {
      grow(type, output.label || output.name);
    }
    return true;
  };
  node.onConnectOutput = function (slot, type, input, target) {
    if (this.outputs[slot]?._append) {
      const widget = target.widgets?.find((w) => w.name === input.name);
      grow(
        type,
        input.name === "position" ? target.title : input.label || input.name,
        widget,
      );
    }
    return true;
  };
  node.onResize = () => appendSocket(node);
  let drag = null,
    animation = 0;
  const rowAt = (pos) => Math.floor((pos[1] - 10) / 28);
  node.onDblClick = function (event, pos) {
    if (!pos || pos[0] < 18 || pos[0] > this.size[0] - 18) return;
    const p = entries(this)[rowAt(pos)];
    if (!p) return;
    // Native editor prompt handles screen placement and text entry.
    const canvas = this.graph?.list_of_graphcanvas?.[0];
    canvas?.prompt(
      "端口名称",
      p.name,
      (value) => {
        if (!value?.trim()) return;
        commit(() => {
          const ps = entries(this);
          ps.find((v) => v.id === p.id).name = value.trim();
          syncPorts(this, ps);
        });
      },
      event,
    );
  };
  node.onMouseDown = function (event, pos) {
    if (
      event.button !== 0 ||
      pos[0] < 22 ||
      pos[0] > this.size[0] - 22 ||
      pos[1] < 10
    )
      return false;
    const ps = entries(this),
      index = rowAt(pos);
    if (!ps[index]) return false;
    drag = {
      id: ps[index].id,
      from: index,
      to: index,
      y: pos[1],
      active: false,
      initial: pos[1],
      positions: new Map(ps.map((p, i) => [p.id, 24 + i * 28])),
    };
    this.captureInput(true);
    return true;
  };
  const animate = () => {
    animation = 0;
    if (!drag?.active) return;
    const ps = entries(node),
      moving = ps.splice(drag.from, 1)[0];
    ps.splice(drag.to, 0, moving);
    let pending = false;
    for (const [i, p] of ps.entries()) {
      const target = p.id === drag.id ? drag.y : 24 + i * 28;
      const previous = drag.positions.get(p.id),
        next = previous + (target - previous) * 0.35;
      drag.positions.set(p.id, Math.abs(target - next) < 0.3 ? target : next);
      pending ||= Math.abs(target - next) >= 0.3;
      for (const s of [...node.inputs, ...node.outputs])
        if (s._portId === p.id) s.pos[1] = drag.positions.get(p.id);
    }
    node.graph?.setDirtyCanvas(true, true);
    if (pending) animation = requestAnimationFrame(animate);
  };
  node.onMouseMove = function (event, pos) {
    if (!drag) return;
    if (Math.abs(pos[1] - drag.initial) > 4) drag.active = true;
    drag.y = Math.max(
      24,
      Math.min(24 + (entries(this).length - 1) * 28, pos[1]),
    );
    drag.to = Math.max(
      0,
      Math.min(entries(this).length - 1, Math.round((drag.y - 24) / 28)),
    );
    if (drag.active && !animation) animation = requestAnimationFrame(animate);
  };
  node.onMouseUp = function () {
    if (!drag) return;
    const state = drag;
    drag = null;
    cancelAnimationFrame(animation);
    animation = 0;
    this.captureInput(false);
    if (state.active && state.from !== state.to)
      commit(() => {
        const ps = entries(this),
          p = ps.splice(state.from, 1)[0];
        ps.splice(state.to, 0, p);
        syncPorts(this, ps);
      });
    else appendSocket(this);
    this.graph?.setDirtyCanvas(true, true);
  };
  node.onKeyDown = function (event) {
    if (event.key !== "Escape" || !drag) return;
    drag = null;
    cancelAnimationFrame(animation);
    animation = 0;
    this.captureInput(false);
    appendSocket(this);
    this.graph?.setDirtyCanvas(true, true);
    return false;
  };
  node.onDrawForeground = function (ctx) {
    if (this.flags.collapsed) return;
    ctx.save();
    ctx.fillStyle = this.bgcolor || "#252a31";
    ctx.fillRect(17, 7, this.size[0] - 34, this.size[1] - 8);
    ctx.font = "13px sans-serif";
    ctx.textBaseline = "middle";
    for (const [i, p] of entries(this).entries()) {
      const y = drag?.active ? drag.positions.get(p.id) : 24 + i * 28;
      if (drag?.active && p.id === drag.id) {
        ctx.fillStyle = "#465970";
        ctx.fillRect(18, y - 12, this.size[0] - 36, 24);
      }
      ctx.fillStyle = "#dce4ed";
      ctx.fillText("⠿  " + p.name, 24, y, this.size[0] - 52);
    }
    ctx.fillStyle = "#9caabd";
    ctx.fillText(
      "拖入连线以添加端口",
      24,
      24 + entries(this).length * 28,
      this.size[0] - 48,
    );
    ctx.restore();
  };
  node.getExtraMenuOptions = function (_, options) {
    if (!isOutput)
      options.push({
        content: "未连接时的默认值…",
        submenu: {
          options: entries(this)
            .filter((p) =>
              ["STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"].includes(p.type),
            )
            .map((p) => ({
              content: p.name,
              callback: () => {
                const value = prompt(
                  p.name +
                    " · 默认值" +
                    (p.options ? " (" + p.options.join(", ") + ")" : ""),
                  String(p.default ?? ""),
                );
                if (value === null) return;
                let parsed =
                  p.type === "BOOLEAN"
                    ? value === "true"
                    : ["INT", "FLOAT"].includes(p.type)
                      ? Number(value)
                      : value;
                if (
                  (p.type === "INT" && !Number.isInteger(parsed)) ||
                  (p.type === "FLOAT" && !Number.isFinite(parsed)) ||
                  (p.options && !p.options.includes(parsed))
                )
                  return;
                commit(() => {
                  const ps = entries(this);
                  ps.find((v) => v.id === p.id).default = parsed;
                  syncPorts(this, ps);
                });
              },
            })),
        },
      });
    options.push({
      content: "删除端口…",
      submenu: {
        options: entries(this).map((p) => ({
          content: p.name,
          callback: () => {
            if (confirm(`删除“${p.name}”及其连线？`))
              commit(() =>
                syncPorts(
                  this,
                  entries(this).filter((v) => v.id !== p.id),
                ),
              );
          },
        })),
      },
    });
  };
  node.color = isOutput ? "#3e4458" : "#354c52";
  node.bgcolor = "#252a31";
}
