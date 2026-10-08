import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { directoryPicker, resolutionControls, rangeControls, restoreValues, persistParameters, socketLabels, MATERIAL_COLORS } from "./controls.js";
import { loraControls, namespaceLabels } from "./settings.js";
import { boundedPreview, requestGuard } from "./ui.js";

const PREFIX = "TuringCanvas";
const isCard = n => (n?.comfyClass ?? n?.type ?? "").startsWith(PREFIX);
const manager = () => app.graph?._nodes?.find(n => n.type === "TuringCanvasSettings");
const value = (n, key) => n.widgets?.find(w => w.name === key)?.value;
let busy = false;
let refreshTimer;
let lastState = {tasks: {}};
let lastPlan = [];
let configuring = false;
let filteredState;
let activeTask = null;
let refreshPending = false;
let refreshAgain = false;
let historyRevision = 0;

function snapshot() {
  if (app.graph._nodes.some(n => (n.properties?.canvasSchemaVersion ?? 1) > 3)) {
    throw new Error("Canvas schema is newer than this plugin; update before editing");
  }
  return {schema_version: 3, nodes: app.graph._nodes.map(n => ({
    id: String(n.id), type: n.comfyClass ?? n.type, mode: n.mode,
    values: {...n.properties?.canvasLegacySelection, ...Object.fromEntries((n.widgets ?? []).filter(w => typeof w.value !== "function" && w.type !== "button" && !w.canvasPreview)
      .map(w => [w.name, w.value]))},
    inputs: Object.fromEntries((n.inputs ?? []).filter(i => i.link != null).map(i => {
      const link = app.graph.links[i.link];
      return [i.name, {node: String(link.origin_id), slot: link.origin_slot,
        ...(n.properties?.canvasPinned?.[i.name] ? {asset: n.properties.canvasPinned[i.name]} : {})}];
    })),
  }))};
}

async function request(path, body) {
  const response = await api.fetchApi(`/turing/canvas/${path}`, {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.error === "string" ? data.error : JSON.stringify(data));
  return data;
}

function report(error) { console.error("Turing Canvas", error); window.alert(error.message ?? String(error)); }

function historyControl(node) {
  const generated = (node.comfyClass ?? node.type) === "TuringCanvasH3";
  const beginRead = requestGuard(node, () => value(manager(), "work_directory"));
  const beginSelection = requestGuard(node, () => value(manager(), "work_directory"));
  let selection = Promise.resolve();
  let widget = generated ? node.addWidget("combo", "历史生成结果", "", async asset => {
    if (!asset) { widget.value = node.properties.canvasAsset || ""; return; }
    const valid = beginSelection(), graph = snapshot();
    const pending = {};
    node.canvasSelectionPending = pending;
    selection = selection.catch(() => {}).then(async () => {
      try { if (!valid()) return; await request("select", {graph, task: String(node.id), asset}); if (valid()) await refresh(); }
      catch (error) { if (valid()) { widget.value = node.properties.canvasAsset || ""; report(error); } }
      finally { if (node.canvasSelectionPending === pending) node.canvasSelectionPending = false; }
    });
    await selection;
  }, {values: [""], serialize: false}) : node.widgets.find(w => w.name === "asset_id");
  if (generated) widget.canvasPreview = true;
  else {
    // Changing a TextWidget's type does not replace its native click handler.
    const original = widget, index = node.widgets.indexOf(original);
    widget = node.addWidget("combo", original.name, original.value, function(...args) {
      original.callback?.apply(this, args); scheduleRefresh();
    }, {...original.options, values: [""], advanced: false});
    node.widgets.splice(node.widgets.indexOf(widget), 1);
    node.widgets.splice(index, 1, widget);
    original.onRemove?.();
    widget.label = "项目素材";
    widget.advanced = false;
  }
  let key;
  node.canvasUpdateHistory = async () => {
    const selected = generated ? node.properties.canvasAsset || "" : widget.value;
    if (generated) widget.value = selected;
    const next = JSON.stringify([value(manager(), "work_directory"), value(node, "filename_prefix"), selected, historyRevision]);
    if (next === key) return;
    key = next;
    const valid = beginRead();
    try {
      const {items} = await request("history", {graph: snapshot(), task: String(node.id)});
      if (key !== next || !valid()) return;
      widget.options.values = [...new Set(["", ...items.map(item => item.id), ...(selected ? [selected] : [])])];
      node.setDirtyCanvas(true, true);
    } catch (error) { if (key === next) key = undefined; }
  };
  const refreshCombo = node.refreshComboInNode;
  node.refreshComboInNode = function(...args) { refreshCombo?.apply(this, args); key = undefined; this.canvasUpdateHistory(); };
}

function preview(node, asset) {
  if (!node.canvasMedia || !manager()) return;
  if (!asset) {
    if (node.canvasMedia.dataset.assetUrl) {
      node.canvasMedia.pause?.(); node.canvasMedia.removeAttribute("src"); node.canvasMedia.removeAttribute("poster");
      delete node.canvasMedia.dataset.assetUrl; node.canvasMedia.load?.(); node.canvasUpdateRange?.(null);
    }
    return;
  }
  const directory = value(manager(), "work_directory");
  const url = api.apiURL(`/turing/canvas/asset?directory=${encodeURIComponent(directory)}&id=${encodeURIComponent(asset)}`);
  if (node.canvasMedia.tagName === "VIDEO") {
    const poster = `${url}&preview=1&start=${Number(value(node, "start_seconds")) || 0}`;
    if (node.canvasMedia.getAttribute("poster") !== poster) node.canvasMedia.poster = poster;
  }
  if (node.canvasMedia.dataset.assetUrl === url) return;
  node.canvasMedia.dataset.assetUrl = url;
  node.canvasMedia.src = node.canvasMedia.tagName === "IMG" ? `${url}&preview=1` : url;
  node.canvasUpdateRange?.(asset);
}

async function refresh() {
  if (!manager()) return;
  if (refreshPending) { refreshAgain = true; return; }
  refreshPending = true;
  try {
  const graph = app.graph, directory = value(manager(), "work_directory");
  const data = await request("state", {graph: snapshot()});
  if (app.graph !== graph || value(manager(), "work_directory") !== directory) return;
  lastState = data.state;
  lastPlan = data.plan;
  for (const node of app.graph._nodes.filter(isCard)) {
    socketLabels(node);
    namespaceLabels(node);
    if (node.canvasMedia?.tagName === "VIDEO") node.canvasMedia.muted = value(node, "include_audio") === false;
    const result = lastState.tasks[String(node.id)];
    if (result) {
      node.properties.canvasAsset = result.asset;
      preview(node, result.asset);
    } else if (["TuringCanvasImage", "TuringCanvasVideo", "TuringCanvasAudio"].includes(node.type)) {
      preview(node, value(node, "asset_id"));
    } else if (node.type === "TuringCanvasH3") {
      delete node.properties.canvasAsset; preview(node, null);
    }
    node.canvasUpdateHistory?.();
    node.canvasStatus = lastPlan.find(p => p.id === String(node.id))?.status ?? "material";
  }
  app.graph.setDirtyCanvas(true, true);
  } finally {
    refreshPending = false;
    if (refreshAgain) { refreshAgain = false; scheduleRefresh(); }
  }
}

function scheduleRefresh() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => { filterTypes(); refresh().catch(() => {}); }, 350);
}

async function waitResult(id) {
  for (;;) {
    await new Promise(resolve => setTimeout(resolve, 1000));
    const response = await api.fetchApi(`/history/${id}`);
    const history = (await response.json())[id];
    if (history) {
      if (history.status?.status_str !== "success") throw new Error(`Task ${id} failed or was interrupted; previous result preserved.`);
      return;
    }
    const queue = await (await api.fetchApi("/queue")).json();
    if (![...queue.queue_running, ...queue.queue_pending].some(q => q[1] === id)) {
      // Check history again to cover the queue->history transition.
      const done = (await (await api.fetchApi(`/history/${id}`)).json())[id];
      if (done?.status?.status_str === "success") return;
      throw new Error("Task failed, was cancelled, or removed from the queue; previous result preserved.");
    }
  }
}

async function runTask(node, graph = snapshot()) {
  node.canvasRunning = true;
  app.graph.setDirtyCanvas(true, true);
  try {
    const result = await request("run", {graph, task: String(node.id), client_id: api.clientId});
    activeTask = {node, promptId: result.prompt_id};
    await waitResult(result.prompt_id);
    historyRevision++;
    node.canvasError = false;
    await refresh();
  } catch (error) { node.canvasError = true; throw error; }
  finally { activeTask = null; node.canvasRunning = false; node.canvasProgress = ""; app.graph.setDirtyCanvas(true, true); }
}

async function generate(node) {
  if (busy) throw new Error("A canvas task is already running. Wait for it to finish.");
  if (app.graph._nodes.some(n => n.canvasSelectionPending)) throw new Error("请等待历史素材选择完成后再生成。");
  busy = true;
  try { await runTask(node); } finally { busy = false; }
}

async function importFile(node, file) {
  const settings = manager();
  if (!settings) throw new Error("Create Canvas Settings first");
  const valid = node.canvasBeginImport();
  const graph = snapshot();
  const form = new FormData();
  form.append("canvas", JSON.stringify({graph, task: String(node.id)}));
  form.append("file", file);
  let asset;
  try {
    const response = await api.fetchApi("/turing/canvas/upload", {method: "POST", body: form});
    asset = await response.json();
    if (!response.ok) throw new Error(asset.error || `HTTP ${response.status}`);
  } catch (uploadError) {
    try {
      asset = await request("import", {graph, task: String(node.id), filename: file.name, size: file.size});
    } catch (copyError) {
      throw new Error(`上传失败：${uploadError.message}\n服务器拷贝失败：${copyError.message}\n浏览器不能提供本机绝对路径；可将文件放到服务器 input 目录，并填写 local_path。`);
    }
  }
  if (!valid()) return;
  historyRevision++;
  node.widgets.find(w => w.name === "asset_id").value = asset.id;
  await refresh();
}

function button(node, name, callback) {
  let pending = false;
  const widget = node.addWidget("button", name, null, async () => {
    if (pending) return;
    pending = true;
    widget.label = `${name} …`;
    app.graph.setDirtyCanvas(true, false);
    try { await callback(); } catch (error) { report(error); }
    finally { pending = false; widget.label = name; app.graph.setDirtyCanvas(true, false); }
  }, {serialize: false});
  return widget;
}

function filterTypes() {
  const active = !!manager();
  if (filteredState === active) return;
  for (const [type, constructor] of Object.entries(LiteGraph.registered_node_types)) {
    if (!Object.hasOwn(constructor, "canvasOriginalSkip")) constructor.canvasOriginalSkip = constructor.skip_list;
    constructor.skip_list = active ? !type.startsWith(PREFIX) : constructor.canvasOriginalSkip;
  }
  // Frontend 1.53 exposes this store through Pinia, but not comfyAPI.
  // Optional bridge: creation/backend validation remain enforced if it moves.
  const store = window.comfyAPI?.nodeDefStore?.useNodeDefStore?.()
    ?? app.extensionManager?._p?._s?.get("nodeDef");
  if (store?.registerNodeDefFilter && filteredState !== active) {
    store.unregisterNodeDefFilter("turing.canvas");
    if (active) store.registerNodeDefFilter({id: "turing.canvas", predicate: def => def.name.startsWith(PREFIX)});
  }
  filteredState = active;
}

app.registerExtension({
  name: "TuringUtils.MaterialCanvas",
  async setup() {
    Object.assign(app.canvas.constructor.link_type_colors, MATERIAL_COLORS);
    api.addEventListener("progress", event => {
      if (!activeTask || (event.detail.prompt_id && event.detail.prompt_id !== activeTask.promptId)) return;
      activeTask.node.canvasProgress = `${event.detail.value}/${event.detail.max}`;
      app.graph.setDirtyCanvas(true, true);
    });
    const queue = app.queuePrompt;
    app.queuePrompt = function (...args) {
      if (!manager() && !app.graph._nodes.some(isCard)) return queue.apply(this, args);
      app.extensionManager?.toast?.add({severity: "info", summary: "Canvas：请使用节点的生成按钮", life: 2500});
      return Promise.resolve();
    };
    const create = LiteGraph.createNode;
    LiteGraph.createNode = function(type, ...args) {
      if (manager() && !type.startsWith(PREFIX) && !configuring) {
        app.extensionManager?.toast?.add({severity: "warn", summary: "Canvas-only workspace", detail: "Ordinary nodes cannot be created here.", life: 3500});
        return null;
      }
      return create.call(this, type, ...args);
    };
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name.startsWith("_TuringCanvas")) nodeType.skip_list = true;
  },
  nodeCreated(node) {
    if (!isCard(node)) return;
    const type = node.comfyClass ?? node.type;
    node.properties ??= {};
    node.canvasBeginImport = requestGuard(node, () => value(manager(), "work_directory"));
    node.canvasBeginEdit = requestGuard(node, () => value(manager(), "work_directory"));
    persistParameters(node);
    namespaceLabels(node);
    if (type === "TuringCanvasH3Settings") loraControls(node, request, scheduleRefresh);
    if (type === "TuringCanvasH3") resolutionControls(node);
    socketLabels(node);
    const changed = node.onWidgetChanged;
    node.onWidgetChanged = function(...args) { changed?.apply(this, args); scheduleRefresh(); };
    const connected = node.onConnectionsChange;
    node.onConnectionsChange = function(...args) { connected?.apply(this, args); socketLabels(this); scheduleRefresh(); };
    const removed = node.onRemoved;
    node.onRemoved = function(...args) { removed?.apply(this, args); this.canvasMedia?.pause?.(); scheduleRefresh(); };
    node.size[0] = 390;
    const draw = node.onDrawForeground;
    node.onDrawForeground = function(ctx, ...args) {
      draw?.call(this, ctx, ...args);
      const status = this.canvasRunning ? "running" : this.canvasError ? "failed" : this.canvasStatus;
      const colors = {running: "#53a7ff", failed: "#ee6565", blocked: "#ee6565", changed: "#e3b64e", upstream: "#e3b64e"};
      if (colors[status]) {
        ctx.save(); ctx.strokeStyle = colors[status]; ctx.lineWidth = 3;
        if (status === "upstream") ctx.setLineDash([8, 5]);
        ctx.strokeRect(0, 0, this.size[0], this.size[1]); ctx.restore();
      }
      const labels = {running: "生成中", failed: "执行失败", blocked: "缺少素材", changed: "待生成", upstream: "上游待更新"};
      if (labels[status] && !this.flags?.collapsed) {
        ctx.save(); ctx.font = "12px sans-serif"; ctx.textAlign = "right";
        const text = `${labels[status]} ${this.canvasProgress ?? ""}`.trim();
        const width = ctx.measureText(text).width + 12;
        ctx.fillStyle = "#202020"; ctx.fillRect(this.size[0] - width - 6, -25, width, 19);
        ctx.fillStyle = colors[status] ?? "#ccc"; ctx.fillText(text, this.size[0] - 12, -11); ctx.restore();
      }
    };
    if (type === "TuringCanvasSettings") {
      const name = `canvas/${crypto.randomUUID()}`;
      for (const key of ["work_directory", "cache_directory"]) {
        const widget = node.widgets.find(w => w.name === key);
        if (widget) widget.value = name;
      }
      const directory = node.widgets.find(w => w.name === "work_directory");
      const browse = button(node, "📁 选择工作目录", () => directoryPicker(node, request));
      node.widgets.splice(node.widgets.indexOf(browse), 1);
      node.widgets.splice(node.widgets.indexOf(directory) + 1, 0, browse);
      button(node, "保存画布", async () => request("save", {graph: snapshot(), workflow: app.graph.serialize()}));
      button(node, "打开已保存画布", async () => {
        if (!window.confirm("Replace the current canvas with the saved project?")) return;
        await app.loadGraphData(await request("load", {graph: snapshot()}));
      });
    } else if (["TuringCanvasImage", "TuringCanvasVideo", "TuringCanvasAudio"].includes(type)) {
      historyControl(node);
      button(node, "导入素材", async () => {
        if (value(node, "local_path")) {
          const valid = node.canvasBeginImport();
          const asset = await request("import", {graph: snapshot(), task: String(node.id)});
          if (!valid()) return;
          historyRevision++;
          node.widgets.find(w => w.name === "asset_id").value = asset.id;
          await refresh();
        } else {
          const input = document.createElement("input"); input.type = "file";
          input.accept = type === "TuringCanvasImage" ? "image/*" : type === "TuringCanvasVideo" ? "video/*" : "audio/*";
          input.onchange = () => input.files[0] && importFile(node, input.files[0]).catch(report);
          input.click();
        }
      });
      node.onDragOver = () => true;
      node.onDragDrop = function(event) {
        if (!event.dataTransfer?.files?.length) return false;
        importFile(node, event.dataTransfer.files[0]).catch(report);
        return true;
      };
    } else if (type === "TuringCanvasH3") {
      const promptButton = button(node, "生成模型提示词", async () => {
        const valid = node.canvasBeginEdit();
        const widget = node.widgets.find(w => w.name === "model_prompt");
        const before = widget.value;
        const result = await request("enhance", {graph: snapshot(), task: String(node.id)});
        if (!valid()) return;
        if (widget.value !== before && !window.confirm("Model prompt was edited during enhancement. Replace it?")) return;
        node.properties.canvasPreviousPrompt = widget.value;
        widget.value = result.model_prompt;
        app.graph.setDirtyCanvas(true, true);
      });
      node.widgets.splice(node.widgets.indexOf(promptButton), 1);
      node.widgets.splice(node.widgets.findIndex(w => w.name === "model_prompt"), 0, promptButton);
      historyControl(node);
    }
    if (!["TuringCanvasSettings", "TuringCanvasH3Settings"].includes(type)) {
      const tag = type === "TuringCanvasImage" ? "img" : type === "TuringCanvasAudio" ? "audio" : "video";
      const media = document.createElement(tag);
      media.style.cssText = "width:100%;height:100%;object-fit:contain;background:#161616";
      if (tag !== "img") { media.controls = true; media.preload = "none"; }
      else { media.loading = "lazy"; media.decoding = "async"; }
      media.addEventListener("dragover", event => event.preventDefault());
      media.addEventListener("drop", event => {
        if (!["TuringCanvasImage", "TuringCanvasVideo", "TuringCanvasAudio"].includes(type)) return;
        event.preventDefault(); event.stopPropagation();
        if (event.dataTransfer.files[0]) importFile(node, event.dataTransfer.files[0]).catch(report);
      });
      const box = document.createElement("div");
      box.style.cssText = "display:flex;flex-direction:column;width:100%;height:100%;min-width:0;min-height:0;max-width:1580px;max-height:900px;overflow:hidden;background:#161616";
      media.style.minHeight = "0"; media.style.minWidth = "0"; media.style.flex = "1 1 0";
      box.append(media);
      boundedPreview(node, box, tag === "audio");
      node.canvasMedia = media;
      node.canvasPreviewBox = box;
    }
    button(node, "恢复上次参数", async () => {
      const valid = node.canvasBeginEdit();
      const result = await request("restore", {graph: snapshot(), task: String(node.id)});
      if (!valid()) return;
      restoreValues(node, result.values);
      scheduleRefresh();
    });
    if (type === "TuringCanvasH3") button(node, "生成", () => generate(node));
    if (["TuringCanvasVideo", "TuringCanvasAudio", "TuringCanvasH3"].includes(type)) rangeControls(node, request, snapshot, scheduleRefresh);
    const oldMenu = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function(_, options) {
      oldMenu?.apply(this, arguments);
      if (this.properties.canvasPreviousPrompt !== undefined) options.push({content: "Undo prompt enhancement", callback: () => {
        this.widgets.find(w => w.name === "model_prompt").value = this.properties.canvasPreviousPrompt;
      }});
      for (const input of this.inputs ?? []) if (input.link != null) {
        options.push({content: `Pin / Follow: ${input.name}`, callback: async () => {
          this.properties.canvasPinned ??= {};
          if (this.properties.canvasPinned[input.name]) delete this.properties.canvasPinned[input.name];
          else {
            const source = app.graph.getNodeById(app.graph.links[input.link].origin_id);
            const asset = value(source, "asset_id") || source.properties.canvasAsset;
            if (!asset) return report(new Error("Source has no published material"));
            this.properties.canvasPinned[input.name] = asset;
          }
          await refresh();
        }});
      }
    };
    const minimum = node.computeSize();
    node.setSize([Math.max(390, minimum[0]), minimum[1]]);
    scheduleRefresh();
  },
  beforeConfigureGraph() { configuring = true; },
  afterConfigureGraph() {
    // v1/v2 split audio output becomes the same video container, consumed by port role.
    for (const node of app.graph._nodes ?? []) {
      if (node.type === "TuringCanvasH3") for (const input of node.inputs ?? []) {
        if (input.name.startsWith("audios.audio_")) input.type = "TURING_CANVAS_AUDIO_ASSET,TURING_CANVAS_VIDEO_ASSET";
      }
      if (!["TuringCanvasVideo", "TuringCanvasH3"].includes(node.type) || node.outputs?.length !== 2) continue;
      for (const id of node.outputs[1].links ?? []) {
        const link = app.graph.links[id];
        if (link) { link.origin_slot = 0; link.type = "TURING_CANVAS_VIDEO_ASSET"; }
        (node.outputs[0].links ??= []).push(id);
      }
      node.outputs[1].links = [];
      node.removeOutput(1);
    }
    configuring = false; filterTypes(); refresh().catch(() => {});
  },
});
