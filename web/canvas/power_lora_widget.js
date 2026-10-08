// Adapted from rgthree-comfy PowerLoraLoaderWidget / RgthreeBaseWidget and
// utils_canvas.js (https://github.com/rgthree/rgthree-comfy).
// Copyright (c) 2023 Regis Gaughan, III (rgthree). MIT; see rgthree-LICENSE.txt.
// Canvas adaptation: one model strength, existing JSON persistence, no model-info
// service. Geometry comes from the node, never the cached widget-width argument.

function fit(ctx, text, width) {
  if (ctx.measureText(text).width <= width) return text;
  let end = text.length;
  while (end && ctx.measureText(text.slice(0, end) + "…").width > width) end--;
  return end ? text.slice(0, end) + "…" : "";
}

function toggle(ctx, x, y, h, value) {
  ctx.save();
  ctx.fillStyle = "#7776";
  ctx.beginPath(); ctx.roundRect(x + 4, y + 4, h * 1.5 - 8, h - 8, h / 2); ctx.fill();
  ctx.fillStyle = value === true ? "#89B" : "#888";
  ctx.beginPath(); ctx.arc(x + h * (value === true ? 1 : value === false ? .5 : .75), y + h / 2, h * .36, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
  return [x, y, h * 1.5, h];
}

const contains = (pos, b) => b && pos[0] >= b[0] && pos[0] <= b[0] + b[2] && pos[1] >= b[1] && pos[1] <= b[1] + b[3];

class PowerWidget {
  constructor(name) {
    this.name = name; this.type = "custom"; this.options = {serialize: false};
    this.serialize = false; this.value = null; this.y = 0; this.last_y = 0;
    this.disabled = false; this.canvasPreview = true; this.canvasLoraRow = true;
    this.hitAreas = {}; this.pressed = null;
  }
  // Match rgthree's fixed native row height, not a width-constrained flex widget.
  computeSize() { return [0, globalThis.LiteGraph?.NODE_WIDGET_HEIGHT ?? 20]; }
  mouse(event, pos, node) {
    if (this.disabled) return false;
    if (["pointerdown", "mousedown"].includes(event.type)) {
      if (event.button != null && event.button !== 0) return false;
      this.pressed = Object.keys(this.hitAreas).find(key => contains(pos, this.hitAreas[key])) ?? null;
      this.startX = pos[0]; this.startStrength = Number(this.entry?.strength ?? 1); this.dragged = false;
      if (this.pressed === "toggle") { this.activate("toggle", event, node); this.pressed = null; }
      return true;
    }
    if (["pointermove", "mousemove"].includes(event.type) && this.pressed?.startsWith("strength")) {
      const dx = pos[0] - this.startX;
      if (Math.abs(dx) >= 2 || this.dragged) {
        this.dragged = true;
        this.setStrength(this.startStrength + dx * .05);
      }
      return true;
    }
    if (["pointerup", "mouseup"].includes(event.type)) {
      const pressed = this.pressed; this.pressed = null;
      if (!this.dragged && pressed && contains(pos, this.hitAreas[pressed])) this.activate(pressed, event, node);
      this.dragged = false;
      return true;
    }
    if (event.type === "pointercancel") this.pressed = null;
    return false;
  }
}

export class PowerLoraRow extends PowerWidget {
  constructor(name, entry, actions) {
    super(name); this.entry = entry; this.actions = actions; this.canvasLoraEntry = true;
  }
  setStrength(value) {
    if (!Number.isFinite(value) || !this.actions.valid()) return;
    this.entry.strength = value; this.actions.changed();
  }
  activate(part, event, node) {
    if (!this.actions.valid()) return;
    if (part === "toggle") { this.entry.on = this.entry.on === false; this.actions.changed(); }
    else if (part === "lora") this.actions.choose(event);
    else if (part === "strengthDec" || part === "strengthInc") {
      this.setStrength(Number((Number(this.entry.strength ?? 1) + (part === "strengthDec" ? -.05 : .05)).toPrecision(15)));
    } else if (part === "strengthVal") this.actions.prompt(event, input => {
      if (String(input ?? "").trim()) this.setStrength(Number(input));
    });
  }
  draw(ctx, node, _cachedWidth, y, height) {
    const width = node.size[0], h = height || this.computeSize()[1];
    // LiteGraph also uses widget.width to reject pointer hits before mouse().
    // Keep that outer hit box consistent with the geometry painted below.
    this.width = width;
    this.last_y = y; this.hitAreas = {};
    ctx.save(); ctx.font = "13px sans-serif"; ctx.textBaseline = "middle";
    ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR; ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.beginPath(); ctx.roundRect(10, y, width - 20, h, h / 2); ctx.fill(); ctx.stroke();
    this.hitAreas.toggle = toggle(ctx, 10, y, h, this.entry.on !== false);
    if (this.entry.on === false) ctx.globalAlpha *= .4;
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    // Same right-aligned number-part layout as rgthree (arrows + value).
    const right = width - 10 - 20 / 3, arrow = 9, gap = 3, number = 48;
    const left = right - arrow * 2 - gap * 2 - number;
    this.hitAreas.strengthDec = [left, y, arrow, h];
    this.hitAreas.strengthVal = [left + arrow + gap, y, number, h];
    this.hitAreas.strengthInc = [right - arrow, y, arrow, h];
    const mid = y + h / 2;
    ctx.beginPath(); ctx.moveTo(left, mid); ctx.lineTo(left + arrow, mid - 5); ctx.lineTo(left + arrow, mid + 5); ctx.fill();
    ctx.beginPath(); ctx.moveTo(right, mid); ctx.lineTo(right - arrow, mid - 5); ctx.lineTo(right - arrow, mid + 5); ctx.fill();
    ctx.textAlign = "center";
    ctx.fillText(fit(ctx, String(Number(this.entry.strength ?? 1)), number), left + arrow + gap + number / 2, mid);
    const x = 10 + h * 1.5 + 10 / 3, labelWidth = Math.max(0, left - 10 / 3 - x);
    this.hitAreas.lora = [x, y, labelWidth, h];
    ctx.textAlign = "left"; ctx.fillText(fit(ctx, this.entry.name || "None", labelWidth), x, mid);
    ctx.restore();
  }
}

export class PowerLoraHeader extends PowerWidget {
  constructor(rows, changed) { super("lora_header"); this.rows = rows; this.changed = changed; }
  activate() {
    const on = !this.rows().every(row => row.on !== false);
    for (const row of this.rows()) row.on = on;
    this.changed();
  }
  draw(ctx, node, _width, y, h) {
    this.width = node.size[0];
    this.last_y = y;
    const rows = this.rows(), all = rows.every(row => row.on !== false), none = rows.every(row => row.on === false);
    ctx.save(); ctx.font = "13px sans-serif"; ctx.textBaseline = "middle";
    this.hitAreas.toggle = toggle(ctx, 10, y, h, all ? true : none ? false : null);
    ctx.globalAlpha *= .55; ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "left"; ctx.fillText("Toggle All", 10 + h * 1.5 + 10 / 3, y + h / 2);
    ctx.textAlign = "right"; ctx.fillText("Strength", node.size[0] - 10 - 20 / 3, y + h / 2);
    ctx.restore();
  }
}
