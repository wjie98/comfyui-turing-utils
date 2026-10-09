import { entries, syncPorts, alignInputs } from "./canvas_ports.js";

// Slot descriptors and link indices stay unchanged until the drag is committed.
export class PortReorder {
  constructor(node, rename) {
    this.node = node;
    this.rename = rename;
    const down = node.onMouseDown,
      move = node.onMouseMove,
      up = node.onMouseUp,
      double = node.onDblClick;
    node.onMouseDown = (event, pos, canvas) => {
      const row = this.hit(pos);
      if (event.button !== 0 || row < 0)
        return down?.call(node, event, pos, canvas);
      this.begin(row);
      return true;
    };
    node.onDblClick = (event, pos, canvas) => {
      const row = this.hit(pos);
      if (row < 0) return double?.call(node, event, pos, canvas);
      this.finish(false);
      rename(entries(node)[row].id);
      return true;
    };
    node.onMouseMove = (event, pos, canvas) => {
      if (!this.drag) return move?.call(node, event, pos, canvas);
      this.drag.target = Math.max(
        0,
        Math.min(
          this.drag.ports.length - 1,
          Math.round((pos[1] - this.drag.top) / LiteGraph.NODE_SLOT_HEIGHT),
        ),
      );
      this.animate();
      return true;
    };
    node.onMouseUp = (event, pos, canvas) => {
      if (!this.drag) return up?.call(node, event, pos, canvas);
      this.finish(true);
      return true;
    };
    const draw = node.onDrawForeground;
    node.onDrawForeground = (ctx, canvas) => {
      draw?.call(node, ctx, canvas);
      if (node.flags.collapsed) return;
      ctx.save();
      ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
      for (let i = 0; i < entries(node).length; i++) {
        const y = node.getConnectionPos(false, i)[1] - node.pos[1];
        ctx.fillText("⋮", node.size[0] / 2 - 3, y + 4);
      }
      if (this.drag) {
        ctx.strokeStyle = LiteGraph.NODE_SELECTED_TITLE_COLOR || "#aaa";
        const y =
          this.drag.top +
          this.drag.target * LiteGraph.NODE_SLOT_HEIGHT -
          LiteGraph.NODE_SLOT_HEIGHT / 2;
        ctx.beginPath();
        ctx.moveTo(20, y);
        ctx.lineTo(node.size[0] - 20, y);
        ctx.stroke();
      }
      ctx.restore();
    };
    const removed = node.onRemoved;
    node.onRemoved = () => {
      this.finish(false);
      removed?.call(node);
    };
  }
  hit(pos) {
    if (
      this.node.flags.collapsed ||
      pos[0] < 24 ||
      pos[0] > this.node.size[0] - 24
    )
      return -1;
    return entries(this.node).findIndex(
      (p, i) =>
        Math.abs(
          pos[1] - (this.node.getConnectionPos(false, i)[1] - this.node.pos[1]),
        ) <
        LiteGraph.NODE_SLOT_HEIGHT / 2,
    );
  }
  begin(index) {
    const node = this.node;
    const ports = entries(node);
    this.drag = {
      ports,
      index,
      target: index,
      top: node.getConnectionPos(false, 0)[1] - node.pos[1],
    };
    node.captureInput(true);
    this.escape = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        this.finish(false);
      }
    };
    document.addEventListener("keydown", this.escape);
  }
  animate() {
    if (this.frame) return;
    const tick = () => {
      this.frame = null;
      if (!this.drag) return;
      const { ports, index, target, top } = this.drag;
      const order = ports.map((p) => p.id);
      order.splice(target, 0, order.splice(index, 1)[0]);
      let moving = false;
      for (const [i, output] of this.node.outputs.entries()) {
        if (output._append) continue;
        const goal =
          top + order.indexOf(output._portId) * LiteGraph.NODE_SLOT_HEIGHT;
        const previous =
          output.pos?.[1] ?? top + i * LiteGraph.NODE_SLOT_HEIGHT;
        const y =
          Math.abs(goal - previous) < 0.5
            ? goal
            : previous + (goal - previous) * 0.35;
        moving ||= y !== goal;
        output.pos = [this.node.size[0], y];
        const input = this.node.inputs.find(
          (s) => s._portId === output._portId,
        );
        if (input) input.pos = [0, y];
      }
      this.node.graph?.setDirtyCanvas(true, true);
      if (moving) this.frame = requestAnimationFrame(tick);
    };
    this.frame = requestAnimationFrame(tick);
  }
  finish(commit) {
    if (!this.drag) return;
    const { ports, index, target } = this.drag;
    this.drag = null;
    cancelAnimationFrame(this.frame);
    this.frame = null;
    document.removeEventListener("keydown", this.escape);
    this.node.captureInput(false);
    if (commit && index !== target) {
      this.node.graph?.beforeChange();
      ports.splice(target, 0, ports.splice(index, 1)[0]);
      syncPorts(this.node, ports);
      this.node.graph?.afterChange();
    } else alignInputs(this.node);
    this.node.graph?.setDirtyCanvas(true, true);
  }
}
