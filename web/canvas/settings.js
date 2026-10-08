import { app } from "../../../scripts/app.js";
import { choiceMenu, requestGuard } from "./ui.js";

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
    const widgets = rows.map((entry, index) => node.addCustomWidget({
      name: `lora_row_${index}`, type: "custom", canvasPreview: true, canvasLoraRow: true,
      options: {serialize: false}, serialize: false, value: null, y: 0, last_y: 0,
      computeLayoutSize: () => ({minWidth: 300, minHeight: 26, maxHeight: 26}),
      draw(ctx, owner, width, y, height) {
        this.last_y = y;
        ctx.save(); ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
        ctx.beginPath(); ctx.roundRect(10, y, width - 20, height, 5); ctx.fill();
        ctx.font = "13px sans-serif"; ctx.textBaseline = "middle";
        ctx.fillStyle = entry.on !== false ? "#72b98b" : "#777";
        ctx.fillText(entry.on !== false ? "●" : "○", 17, y + height / 2);
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.save(); ctx.beginPath(); ctx.rect(38, y, Math.max(0, width - 180), height); ctx.clip();
        ctx.fillText(entry.name || "选择 LoRA", 38, y + height / 2); ctx.restore();
        ctx.textAlign = "center";
        ctx.fillText("▾", width - 133, y + height / 2);
        ctx.fillText("−", width - 112, y + height / 2);
        ctx.fillText(Number(entry.strength ?? 1).toFixed(2), width - 78, y + height / 2);
        ctx.fillText("+", width - 45, y + height / 2);
        ctx.fillText("×", width - 20, y + height / 2); ctx.restore();
      },
      mouse(event, pos) {
        if (!["pointerdown", "mousedown"].includes(event.type)) return false;
        const x = pos[0], width = node.size[0];
        if (x < 35) { entry.on = entry.on === false; commit(); }
        else if (x < width - 122) select(event, name => { if (rows.includes(entry)) { entry.name = name; commit(); } });
        else if (x > width - 32) { rows.splice(index, 1); commit(); rebuild(); }
        else if (x < width - 98 || x > width - 57) {
          entry.strength = Math.round((Number(entry.strength ?? 1) + (x < width - 98 ? -.05 : .05)) * 100) / 100; commit();
        } else app.canvas.prompt("LoRA strength", String(entry.strength ?? 1), input => {
          const strength = Number(input); if (rows.includes(entry) && input?.trim() && Number.isFinite(strength)) { entry.strength = strength; commit(); }
        }, event);
        return true;
      },
    }));
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
  const callback = storage.callback;
  storage.callback = function(...args) { callback?.apply(this, args); rebuild(); };
  const draw = node.onDrawForeground;
  node.onDrawForeground = function(...args) { draw?.apply(this, args); if (current !== storage.value) rebuild(); };
  rebuild();
}
