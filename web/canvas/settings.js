import { app } from "../../../scripts/app.js";

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

export function loraControls(node, request, changed) {
  const value = node.widgets.find(w => w.name === "loras");
  value.type = "hidden"; value.computeSize = () => [0, -4];
  const box = document.createElement("div");
  box.style.cssText = "display:flex;flex-direction:column;gap:4px;width:100%;color:var(--input-text)";
  let names = [], current = null;
  const commit = rows => { value.value = JSON.stringify(rows); changed(); app.graph.setDirtyCanvas(true, true); };
  const render = () => {
    let rows;
    try { rows = JSON.parse(value.value || "[]"); if (!Array.isArray(rows)) throw Error("LoRA list required"); }
    catch (error) { box.textContent = error.message; return; }
    current = value.value;
    box.replaceChildren();
    rows.forEach((entry, index) => {
      const row = document.createElement("div"); row.style.cssText = "display:flex;gap:4px;min-width:0";
      const on = document.createElement("input"); on.type = "checkbox"; on.checked = entry.on !== false; on.title = "启用 LoRA";
      const model = document.createElement("select"); model.style.cssText = "min-width:0;flex:1";
      for (const name of new Set([entry.name || "", ...names])) { const o = document.createElement("option"); o.value = name; o.textContent = name || "选择 LoRA"; model.append(o); }
      model.value = entry.name || "";
      const strength = document.createElement("input"); strength.type = "number"; strength.step = "0.05"; strength.value = entry.strength ?? 1; strength.style.width = "60px"; strength.title = "LoRA strength";
      const remove = document.createElement("button"); remove.textContent = "×"; remove.title = "删除 LoRA";
      const update = () => { rows[index] = {on: on.checked, name: model.value, strength: Number(strength.value)}; commit(rows); };
      on.onchange = model.onchange = strength.onchange = update;
      remove.onclick = () => { rows.splice(index, 1); commit(rows); render(); node.setSize(node.computeSize()); };
      row.append(on, model, strength, remove); box.append(row);
    });
    const add = document.createElement("button"); add.textContent = "+ 添加 LoRA";
    add.onclick = async () => {
      try { if (!names.length) names = (await request("loras", {})).names; rows.push({on: true, name: names[0] || "", strength: 1}); commit(rows); render(); node.setSize(node.computeSize()); }
      catch (error) { add.textContent = error.message; }
    };
    box.append(add);
  };
  const ui = node.addDOMWidget("lora_stack", "canvas_loras", box, {serialize: false, hideOnZoom: true});
  ui.canvasPreview = true; ui.computeSize = () => [350, Math.max(1, box.children.length) * 30];
  node.widgets.splice(node.widgets.indexOf(ui), 1);
  node.widgets.splice(node.widgets.indexOf(value) + 1, 0, ui);
  const callback = value.callback;
  value.callback = function(...args) { callback?.apply(this, args); render(); };
  const draw = node.onDrawForeground;
  node.onDrawForeground = function(...args) { draw?.apply(this, args); if (current !== value.value) render(); };
  render();
  request("loras", {}).then(result => { names = result.names; render(); }).catch(() => {});
}
