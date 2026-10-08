import { app } from "../../../scripts/app.js";
import { choiceMenu, requestGuard } from "./ui.js";
import { PowerLoraRow, PowerLoraHeader } from "./power_lora_widget.js";

export function namespaceLabels(node) {
  for (const w of node.widgets ?? []) {
    const name = w.name;
    const group = name.startsWith("chat_") ? "Prompt Chat" : name.startsWith("strategy") ? "Attention Strategy" :
      ["attention", "force_int8_gemm", "dit", "clip"].includes(name) ? "ConvRot Loader" :
      ["video_vae", "audio_vae"].includes(name) ? "VAE Loader" : name.startsWith("shift_") ? "H3 Sigma Shift" :
      ["steps", "scheduler", "sampler_name", "refiner", "denoise"].includes(name) ? "Sampling" :
      ["aspect_ratio", "megapixels", "width", "height"].includes(name) ? "Resolution Selector" : "";
    if (group) w.label = `${group} / ${name.replace(/^chat_|^strategy\./, "")}`;
  }
}

// rgthree uses Canvas row widgets and this LiteGraph menu, not HTML selects.
export function loraControls(node, request, changed) {
  const storage = node.widgets.find(w => w.name === "loras");
  storage.hidden = true; storage.options.hidden = true;
  storage.type = "hidden"; storage.computeSize = () => [0, -4];
  let rows = [], current;
  const beginSelection = requestGuard(node);
  const commit = () => { storage.value = JSON.stringify(rows); current = storage.value; changed(); node.setDirtyCanvas(true, true); };
  const select = async (event, callback) => {
    const valid = beginSelection();
    try {
      const {names} = await request("loras", {});
      if (valid()) choiceMenu(names, event, "选择 LoRA", name => { if (valid()) callback(name); });
    } catch (error) { app.extensionManager?.toast?.add({severity: "error", summary: error.message}); }
  };
  const rebuild = () => {
    current = storage.value;
    try {
      const parsed = JSON.parse(storage.value || "[]");
      if (!Array.isArray(parsed) || parsed.some(row => !row || typeof row.name !== "string" || !Number.isFinite(Number(row.strength ?? 1)))) throw Error("Invalid LoRA list");
      rows = parsed;
    }
    catch (error) { app.extensionManager?.toast?.add({severity: "error", summary: error.message}); return; }
    current = storage.value;
    node.widgets = node.widgets.filter(w => !w.canvasLoraRow);
    const widgets = rows.map((entry, index) => node.addCustomWidget(new PowerLoraRow(`lora_row_${index}`, entry, {
      valid: () => rows.includes(entry), changed: commit,
      choose: event => select(event, name => { if (rows.includes(entry)) { entry.name = name; commit(); } }),
      prompt: (event, callback) => app.canvas.prompt("LoRA strength", String(entry.strength ?? 1), callback, event),
    })));
    if (rows.length) widgets.unshift(node.addCustomWidget(new PowerLoraHeader(() => rows, commit)));
    for (const row of widgets) node.widgets.splice(node.widgets.indexOf(row), 1);
    const add = node.addWidget("button", "+ 添加 LoRA", null, (...args) => {
      const event = args.find(arg => arg && typeof arg.clientX === "number");
      select(event, name => { rows.push({on: true, name, strength: 1}); commit(); rebuild(); });
    }, {serialize: false});
    add.canvasLoraRow = true; add.canvasPreview = true;
    node.widgets.splice(node.widgets.indexOf(add), 1);
    node.widgets.splice(node.widgets.indexOf(storage) + 1, 0, ...widgets, add);
    const minimum = node.computeSize();
    node.setSize([Math.max(node.size[0], minimum[0]), Math.max(node.size[1], minimum[1])]);
    node.setDirtyCanvas(true, true);
  };
  // Match Power LoRA's row context menu rather than squeezing a delete button
  // into the strength control. Use the very same painted row bounds.
  const getSlot = node.getSlotInPosition;
  const slotMenu = node.getSlotMenuOptions;
  node.getSlotInPosition = function(x, y, ...args) {
    const slot = getSlot?.call(this, x, y, ...args);
    if (slot) { this.getSlotMenuOptions = slotMenu; return slot; }
    const widget = this.widgets.find(w => w.canvasLoraEntry && x >= this.pos[0] + 10 && x <= this.pos[0] + this.size[0] - 10 && y >= this.pos[1] + w.last_y && y <= this.pos[1] + w.last_y + w.computeSize()[1]);
    this.getSlotMenuOptions = widget ? loraMenu : slotMenu;
    return widget ? {widget, output: {type: "LORA WIDGET"}} : slot;
  };
  const loraMenu = function(slot) {
    if (!slot.widget?.canvasLoraEntry) return slotMenu?.call(this, slot);
    const entry = slot.widget.entry, index = rows.indexOf(entry);
    const update = fn => () => { if (!rows.includes(entry)) return; fn(); commit(); rebuild(); };
    return [
      {content: entry.on === false ? "Toggle On" : "Toggle Off", callback: update(() => { entry.on = entry.on === false; })},
      {content: "Move Up", disabled: index <= 0, callback: update(() => { const i = rows.indexOf(entry); if (i > 0) [rows[i-1], rows[i]] = [rows[i], rows[i-1]]; })},
      {content: "Move Down", disabled: index >= rows.length - 1, callback: update(() => { const i = rows.indexOf(entry); if (i < rows.length - 1) [rows[i+1], rows[i]] = [rows[i], rows[i+1]]; })},
      {content: "Remove LoRA", callback: update(() => rows.splice(rows.indexOf(entry), 1))},
    ];
  };
  node.getSlotMenuOptions = loraMenu;
  const callback = storage.callback;
  storage.callback = function(...args) { callback?.apply(this, args); rebuild(); };
  const draw = node.onDrawForeground;
  node.onDrawForeground = function(...args) { draw?.apply(this, args); if (current !== storage.value) rebuild(); };
  rebuild();
}
