import { app } from "../../../scripts/app.js";

const widget = (node, key) => node.widgets?.find(w => w.name === key);
export const MATERIAL_COLORS = {TURING_CANVAS_IMAGE_ASSET: "#72b98b", TURING_CANVAS_VIDEO_ASSET: "#55b9c5", TURING_CANVAS_AUDIO_ASSET: "#d49bdf"};
const roundEven = n => Math.abs(n % 1) === 0.5 ? 2 * Math.round(n / 2) : Math.round(n);
const align = n => Math.max(32, roundEven(Number(n) / 32) * 32);

export function restoreValues(node, values) {
  // Set DynamicCombo selectors before their generated child widgets.
  for (const [name, value] of Object.entries(values).sort(([a], [b]) => a.split(".").length - b.split(".").length)) {
    const item = widget(node, name);
    if (!item || item.type === "button" || item.canvasPreview) continue;
    item.value = value;
    item.callback?.(value);
  }
  node.canvasSyncResolution?.("width");
  node.canvasSyncRange?.();
  app.graph.setDirtyCanvas(true, true);
}

export function persistParameters(node) {
  // Native number widgets derive precision from drag step, not rounding granularity.
  // Keep convenient 0.1-second dragging without rounding old frame times to tenths.
  for (const name of ["duration", "start_seconds", "end_seconds"]) {
    const item = widget(node, name);
    if (item) item.options.precision = 3;
  }
  const limit = widget(node, "max_megapixels");
  if (limit) {
    const callback = limit.callback;
    limit.callback = function(...args) {
      callback?.apply(this, args);
      if (!node.canvasRestoring && node.properties.canvasLegacySelection) {
        delete node.properties.canvasLegacySelection.custom_width;
        delete node.properties.canvasLegacySelection.custom_height;
      }
    };
  }
  const serialize = node.onSerialize;
  node.onSerialize = function(info) {
    serialize?.call(this, info);
    info.properties ??= {};
    if ((info.properties.canvasSchemaVersion ?? 1) > 3) return;
    info.properties.canvasSchemaVersion = 3;
    info.properties.canvasParameters = Object.fromEntries(this.widgets.filter(w =>
      w.type !== "button" && !w.canvasPreview && typeof w.value !== "function").map(w => [w.name, w.value]));
  };
  const configure = node.onConfigure;
  node.onConfigure = function(info) {
    configure?.call(this, info);
    const version = info.properties?.canvasSchemaVersion ?? 1;
    if (version > 3) {
      this.canvasError = true;
      throw new Error("Canvas schema is newer than this plugin; update before editing");
    }
    let values = info.properties?.canvasParameters ?? info.widgets_values_named;
    if (!values && Array.isArray(info.widgets_values)) {
      const keys = {
        TuringCanvasSettings: ["work_directory", "cache_directory", "import_mode", "max_megapixels"],
        TuringCanvasImage: ["asset_id", "local_path"],
        TuringCanvasVideo: ["asset_id", "local_path", "start_seconds", "duration_seconds", "force_rate", "custom_width", "custom_height", "skip_first_frames", "frame_load_cap", "select_every_nth"],
        TuringCanvasAudio: ["asset_id", "local_path", "start_seconds", "duration_seconds"],
        TuringCanvasH3: ["user_prompt", "model_prompt", "width", "height", "frames", "denoise", "filename_prefix", "preserve_audio"],
      }[this.comfyClass ?? this.type];
      if (keys) values = Object.fromEntries(keys.map((k, i) => [k, info.widgets_values[i]]).filter(([, v]) => v !== undefined));
      else if (version === 1) {
        // The old H3 settings list contains a variable number of Sol controls.
        const keys = ["dit", "clip", "video_vae", "audio_vae", "loras", "attention", "sol"];
        if (info.widgets_values[6] === "enabled") keys.push("sol.routing_threshold", "sol.dense_prefix_steps", "sol.dense_suffix_steps", "sol.dense_prefix_layers", "sol.dense_suffix_layers", "sol.sparse_reference_image", "sol.sparse_reference_video", "sol.sparse_reference_audio");
        keys.push("steps", "sigmas", "shift_video", "shift_audio", "chat_url", "chat_model", "chat_api_key_env", "chat_system_prompt");
        values = Object.fromEntries(keys.map((k, i) => [k, info.widgets_values[i]]).filter(([, v]) => v !== undefined));
      }
    }
    if (values) {
      values = {...values};
      // Native positional restore runs before this hook; never let removed fields
      // populate newly added sampler or trim widgets in old workflows.
      if (version < 3) {
        for (const [key, fallback] of Object.entries({sampler_name: "euler", scheduler: "simple", refiner: true, start_seconds: 0, end_seconds: 0})) {
          if (values[key] === undefined && !(key === "end_seconds" && values.duration_seconds !== undefined)) values[key] = fallback;
        }
      }
      if (values.frames !== undefined && values.duration === undefined) values.duration = values.frames / 24;
      if (values.duration_seconds !== undefined && values.end_seconds === undefined) {
        values.end_seconds = values.duration_seconds ? Number(values.start_seconds || 0) + Number(values.duration_seconds) : 0;
        // Preserve v1 frame-based selections until the user explicitly edits the range.
        this.properties.canvasLegacySelection = Object.fromEntries(
          ["force_rate", "skip_first_frames", "frame_load_cap", "select_every_nth", "custom_width", "custom_height"]
            .filter(key => values[key] !== undefined).map(key => [key, values[key]]));
      }
      if (values.sol !== undefined && values.strategy === undefined) {
        values.strategy = values.sol === "enabled" ? "sol" : "disabled";
        for (const [key, val] of Object.entries(values)) if (key.startsWith("sol.")) {
          const name = key.slice(4);
          values[`strategy.${name.startsWith("sparse_reference_") ? "prefix_policy." : ""}${name}`] = val;
        }
      }
      this.canvasRestoring = true;
      try { restoreValues(this, values); } finally { this.canvasRestoring = false; }
    }
    socketLabels(this);
  };
}

export function socketLabels(node) {
  for (const slot of [...(node.inputs ?? []), ...(node.outputs ?? [])]) {
    if (MATERIAL_COLORS[slot.type]) slot.color_on = slot.color_off = MATERIAL_COLORS[slot.type];
    const match = /^(image|video|audio)s\.\1_(\d+)$/.exec(slot.name);
    if (match) {
      slot.label = `${match[1]} ${Number(match[2]) + 1}`;
      slot.color_on = slot.color_off = MATERIAL_COLORS[`TURING_CANVAS_${match[1].toUpperCase()}_ASSET`];
    }
  }
  const type = node.comfyClass ?? node.type;
  if (type === "TuringCanvasImage" && node.outputs?.[0]) node.outputs[0].label = "image";
  if (type === "TuringCanvasAudio" && node.outputs?.[0]) node.outputs[0].label = "audio";
  if (type === "TuringCanvasVideo") {
    const audio = widget(node, "include_audio")?.value !== false;
    if (node.outputs?.[0]) node.outputs[0].label = audio ? "video (AV)" : "video";
  }
}

export function resolutionControls(node) {
  let syncing = false;
  node.canvasSyncResolution = key => {
    if (syncing) return;
    syncing = true;
    try {
      const w = widget(node, "width"), h = widget(node, "height"), aspect = widget(node, "aspect_ratio"), mp = widget(node, "megapixels");
      if (["aspect_ratio", "megapixels"].includes(key)) {
        const ratio = /^(\d+):(\d+)/.exec(aspect.value);
        const a = ratio ? +ratio[1] : +w.value, b = ratio ? +ratio[2] : +h.value;
        const scale = Math.sqrt(Number(mp.value) * 1024 * 1024 / (a * b));
        w.value = align(a * scale); h.value = align(b * scale);
      } else {
        w.value = align(w.value); h.value = align(h.value);
        mp.value = w.value * h.value / (1024 * 1024);
        const options = typeof aspect.options.values === "function" ? aspect.options.values() : aspect.options.values;
        aspect.value = options.find(option => {
          const ratio = /^(\d+):(\d+)/.exec(option);
          return ratio && Number(ratio[1]) * h.value === Number(ratio[2]) * w.value;
        }) ?? "Custom";
      }
      aspect.label = `aspect_ratio (${w.value}:${h.value})`;
    } finally { syncing = false; }
  };
  for (const name of ["aspect_ratio", "megapixels", "width", "height"]) {
    const item = widget(node, name), callback = item.callback;
    item.callback = function(...args) { callback?.apply(this, args); node.canvasSyncResolution(name); };
  }
}

export function rangeControls(node, request, snapshot, changed) {
  const clearLegacy = () => {
    if (node.canvasRestoring) return;
    for (const key of ["force_rate", "skip_first_frames", "frame_load_cap", "select_every_nth"]) {
      if (node.properties.canvasLegacySelection) delete node.properties.canvasLegacySelection[key];
    }
  };
  const row = document.createElement("div");
  row.style.cssText = "flex:0 0 64px;padding:4px 10px;box-sizing:border-box;color:var(--input-text);font:12px sans-serif";
  const media = node.canvasMedia;
  media.controls = false;
  const toolbar = document.createElement("div"); toolbar.style.cssText = "display:flex;align-items:center;gap:8px;height:26px";
  const play = document.createElement("button"); play.textContent = "▶";
  const label = document.createElement("span");
  const bar = document.createElement("div"); bar.style.cssText = "position:relative;height:24px;touch-action:none;cursor:pointer";
  const track = document.createElement("div"); track.style.cssText = "position:absolute;left:0;right:0;top:9px;height:6px;background:#444;border-radius:3px";
  const selected = document.createElement("div"); selected.style.cssText = "position:absolute;top:9px;height:6px;background:#55b9c5";
  const cursor = document.createElement("div"); cursor.style.cssText = "position:absolute;top:3px;width:2px;height:18px;background:white;pointer-events:none";
  bar.append(track, selected, cursor);
  toolbar.append(play, label); row.append(toolbar, bar);
  let duration = 0;
  let stopTimer;
  const bounds = () => [Math.min(Number(widget(node, "start_seconds").value), duration), Math.min(Number(widget(node, "end_seconds").value) || duration, duration)];
  const seek = value => { if (media.readyState) media.currentTime = value; };
  const scheduleStop = () => {
    clearTimeout(stopTimer);
    if (!media.paused && duration) stopTimer = setTimeout(() => media.pause(), Math.max(0, (bounds()[1] - media.currentTime) / media.playbackRate * 1000));
  };
  play.onclick = async () => { if (!media.paused) media.pause(); else { try { await media.play(); } catch (error) { row.title = error.message; } } };
  media.addEventListener("play", () => { const [a, b] = bounds(); if (media.currentTime < a || media.currentTime >= b) seek(a); play.textContent = "❚❚"; });
  media.addEventListener("pause", () => { clearTimeout(stopTimer); play.textContent = "▶"; });
  for (const event of ["playing", "seeked", "ratechange"]) media.addEventListener(event, scheduleStop);
  media.addEventListener("waiting", () => clearTimeout(stopTimer));
  const removed = node.onRemoved;
  node.onRemoved = function(...args) { clearTimeout(stopTimer); removed?.apply(this, args); };
  const clampPlayback = () => {
    const [a, b] = bounds();
    if (duration && media.currentTime < a) seek(a);
    if (duration && media.currentTime >= b) { media.pause(); if (media.currentTime > b) seek(b); }
    cursor.style.left = `${duration ? media.currentTime / duration * 100 : 0}%`;
  };
  media.addEventListener("loadedmetadata", () => { duration = media.duration; node.canvasSyncRange(); clampPlayback(); });
  media.addEventListener("seeking", clampPlayback);
  media.addEventListener("timeupdate", clampPlayback);
  bar.onpointerdown = event => {
    if (event.target !== bar && event.target !== track && event.target !== selected) return;
    const rect = bar.getBoundingClientRect(), [a, b] = bounds();
    seek(Math.max(a, Math.min(b, (event.clientX - rect.left) / rect.width * duration)));
  };
  const inputs = [];
  for (const name of ["start_seconds", "end_seconds"]) {
    const input = document.createElement("button"); input.textContent = name === "start_seconds" ? "[" : "]";
    input.style.cssText = "position:absolute;top:0;transform:translateX(-50%);width:16px;height:24px;padding:0;cursor:ew-resize;touch-action:none";
    input.disabled = true;
    input.setAttribute("aria-label", name === "start_seconds" ? "剪切起点" : "剪切终点");
    const update = seconds => {
      clearLegacy();
      const start = widget(node, "start_seconds"), end = widget(node, "end_seconds");
      widget(node, name).value = seconds;
      const last = Number(end.value) || duration;
      if (start.value >= last) widget(node, name).value = name === "start_seconds" ? Math.max(0, last - .01) : Number(start.value) + .01;
      node.canvasSyncRange();
      // Seek only a media stream already opened by the user; dragging alone must not download it.
      if (media.readyState > 0) seek(Number(widget(node, name).value));
    };
    input.onpointerdown = event => { event.stopPropagation(); media.pause(); input.setPointerCapture(event.pointerId); };
    input.onpointermove = event => {
      if (!input.hasPointerCapture(event.pointerId)) return;
      const rect = bar.getBoundingClientRect();
      update(Math.round(Math.max(0, Math.min(duration, (event.clientX - rect.left) / rect.width * duration)) * 1000) / 1000);
    };
    input.onpointerup = event => { if (input.hasPointerCapture(event.pointerId)) input.releasePointerCapture(event.pointerId); changed(); };
    input.onkeydown = event => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault(); update(Math.max(0, Math.min(duration, Number(widget(node, name).value) + (event.key === "ArrowRight" ? .01 : -.01)))); changed();
    };
    bar.append(input); inputs.push(input);
  }
  node.canvasPreviewBox.append(row);
  node.canvasSyncRange = () => {
    const [a, b] = bounds();
    inputs.forEach((input, i) => { input.style.left = `${duration ? (i ? b : a) / duration * 100 : 0}%`; input.title = `${i ? b : a}s`; });
    selected.style.left = `${duration ? a / duration * 100 : 0}%`;
    selected.style.width = `${duration ? (b - a) / duration * 100 : 100}%`;
    label.textContent = `${a.toFixed(2)} – ${b.toFixed(2)} s · ${(b - a).toFixed(2)} s`;
  };
  let revision = 0;
  node.canvasUpdateRange = async asset => {
    const current = ++revision;
    inputs.forEach(input => { input.disabled = true; });
    try {
      const metadata = await request("metadata", {graph: snapshot(), asset});
      if (current !== revision) return;
      duration = Number(metadata.duration);
      if (!(duration > 0)) return;
      inputs.forEach(input => { input.disabled = false; });
      node.canvasSyncRange();
    } catch (error) { row.title = error.message; }
  };
  for (const name of ["start_seconds", "end_seconds"]) {
    const item = widget(node, name), callback = item.callback;
    item.callback = function(...args) { callback?.apply(this, args); clearLegacy(); node.canvasSyncRange(); changed(); };
  }
}

export async function directoryPicker(node, request) {
  const dialog = document.createElement("dialog");
  dialog.style.cssText = "width:440px;max-width:85vw;background:var(--comfy-menu-bg);color:var(--input-text);padding:16px";
  const heading = document.createElement("div"), list = document.createElement("select"), status = document.createElement("div");
  list.size = 12; list.style.width = "100%";
  const actions = document.createElement("div"); actions.style.cssText = "display:flex;gap:8px;margin-top:12px";
  let path = "", pending = false;
  const render = async () => {
    pending = true; status.textContent = "加载目录…"; list.disabled = true;
    try {
      const data = await request("directories", {path});
      heading.textContent = `output/${path}`;
      list.replaceChildren(...data.directories.map(name => { const item = document.createElement("option"); item.value = name; item.textContent = name; return item; }));
      list.selectedIndex = -1; status.textContent = "双击进入目录；选择目录后可清理空目录";
    } catch (error) { status.textContent = error.message; }
    finally { pending = false; list.disabled = false; }
  };
  const action = (text, fn) => {
    const btn = document.createElement("button"); btn.textContent = text;
    btn.onclick = async () => { if (pending) return; try { await fn(); } catch (error) { status.textContent = error.message; } };
    actions.append(btn);
  };
  list.ondblclick = async () => { if (!list.value || pending) return; path = [path, list.value].filter(Boolean).join("/"); await render(); };
  action("上一级", async () => { path = path.split("/").slice(0, -1).join("/"); await render(); });
  action("选择", () => {
    const selected = [path, list.value].filter(Boolean).join("/");
    if (!selected) throw new Error("请选择 output 下的项目目录");
    widget(node, "work_directory").value = selected; widget(node, "work_directory").callback?.(selected); dialog.close();
  });
  action("新目录", () => {
    const name = window.prompt("目录名（首次保存或导入时创建）");
    if (!name) return;
    if (/[\\/]/.test(name) || [".", ".."].includes(name)) throw new Error("请输入单个目录名");
    const selected = [path, name].filter(Boolean).join("/");
    widget(node, "work_directory").value = selected; widget(node, "work_directory").callback?.(selected); dialog.close();
  });
  action("删除空目录", async () => {
    const selected = [path, list.value].filter(Boolean).join("/");
    if (!selected || !window.confirm(`仅删除递归为空的目录 output/${selected}？`)) return;
    pending = true;
    try { await request("remove-directory", {path: selected}); if (!list.value) path = path.split("/").slice(0, -1).join("/"); }
    finally { pending = false; }
    await render();
  });
  action("取消", () => dialog.close());
  dialog.onclose = () => dialog.remove();
  dialog.append(heading, list, status, actions); document.body.append(dialog); dialog.showModal();
  await render();
}
