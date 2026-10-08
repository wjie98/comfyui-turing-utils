import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const PREFIX = "TuringCanvas";
const isCard = n => (n?.comfyClass ?? n?.type ?? "").startsWith(PREFIX);
const manager = () => app.graph?._nodes?.find(n => n.type === "TuringCanvasSettings");
const value = (n, key) => n.widgets?.find(w => w.name === key)?.value;
let armed = false;
let busy = false;
let refreshTimer;
let lastState = {tasks: {}};
let lastPlan = [];
let configuring = false;
let filteredState;
let activeTask = null;
let refreshPending = false;

function snapshot() {
  return {nodes: app.graph._nodes.map(n => ({
    id: String(n.id), type: n.comfyClass ?? n.type, mode: n.mode,
    values: Object.fromEntries((n.widgets ?? []).filter(w => typeof w.value !== "function" && w.type !== "button" && !w.canvasPreview)
      .map(w => [w.name, w.value])),
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

function choose(title, items) {
  return new Promise(resolve => {
    const dialog = document.createElement("dialog");
    dialog.style.cssText = "min-width:360px;max-width:80vw;background:var(--comfy-menu-bg);color:var(--input-text);border:1px solid #555;padding:20px";
    const heading = document.createElement("h3"); heading.textContent = title;
    const select = document.createElement("select"); select.size = 12; select.style.cssText = "width:100%;min-height:220px";
    for (const item of items) { const option = document.createElement("option"); option.value = item.id; option.textContent = item.name; select.append(option); }
    const ok = document.createElement("button"); ok.textContent = "选择";
    const cancel = document.createElement("button"); cancel.textContent = "取消";
    let answer = null;
    ok.onclick = () => { answer = select.value || null; dialog.close(); };
    select.ondblclick = ok.onclick;
    cancel.onclick = () => dialog.close();
    dialog.onclose = () => { dialog.remove(); resolve(answer); };
    dialog.append(heading, select, ok, cancel); document.body.append(dialog); dialog.showModal();
  });
}

async function history(node) {
  const {items} = await request("history", {graph: snapshot(), task: String(node.id)});
  const asset = await choose(node.type === "TuringCanvasH3" ? "历史生成结果" : "项目素材", items);
  if (!asset) return;
  if (node.type === "TuringCanvasH3") await request("select", {graph: snapshot(), task: String(node.id), asset});
  else node.widgets.find(w => w.name === "asset_id").value = asset;
  await refresh();
}

async function directoryPicker(node) {
  let path = "";
  for (;;) {
    const data = await request("directories", {path});
    const selected = await choose(`服务器 output/${path}`, [
      ...(path ? [{id: "select", name: "使用当前目录"}, {id: "up", name: ".."}] : []),
      {id: "new", name: "新建项目目录…"}, ...data.directories.map(name => ({id: "dir:" + name, name}))]);
    if (!selected) return;
    if (selected === "up") { path = path.split("/").slice(0, -1).join("/"); continue; }
    if (selected.startsWith("dir:")) { path = [path, selected.slice(4)].filter(Boolean).join("/"); continue; }
    if (selected === "new") {
      const name = window.prompt("新项目目录名");
      if (!name) return;
      if (/[\\/]/.test(name) || [".", ".."].includes(name)) throw new Error("请输入目录名，不是路径");
      path = [path, name].filter(Boolean).join("/");
    }
    node.widgets.find(w => w.name === "work_directory").value = path;
    await refresh(); return;
  }
}

function preview(node, asset) {
  if (!node.canvasMedia || !asset || !manager()) return;
  const directory = value(manager(), "work_directory");
  const url = api.apiURL(`/turing/canvas/asset?directory=${encodeURIComponent(directory)}&id=${encodeURIComponent(asset)}`);
  if (node.canvasMedia.dataset.assetUrl === url) return;
  node.canvasMedia.dataset.assetUrl = url;
  node.canvasMedia.src = node.canvasMedia.tagName === "IMG" ? `${url}&preview=1` : url;
  if (node.canvasMedia.tagName === "VIDEO") node.canvasMedia.poster = `${url}&preview=1`;
}

async function refresh() {
  if (!manager()) { armed = false; return; }
  if (refreshPending) return;
  refreshPending = true;
  try {
  const data = await request("state", {graph: snapshot()});
  lastState = data.state;
  lastPlan = data.plan;
  for (const node of app.graph._nodes.filter(isCard)) {
    const result = lastState.tasks[String(node.id)];
    if (result) {
      node.properties.canvasAsset = result.asset;
      preview(node, result.asset);
    } else if (["TuringCanvasImage", "TuringCanvasVideo", "TuringCanvasAudio"].includes(node.type)) {
      preview(node, value(node, "asset_id"));
    }
    node.canvasStatus = lastPlan.find(p => p.id === String(node.id))?.status ?? "material";
  }
  app.graph.setDirtyCanvas(true, true);
  } finally { refreshPending = false; }
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
    node.canvasError = false;
    await refresh();
  } catch (error) { node.canvasError = true; throw error; }
  finally { activeTask = null; node.canvasRunning = false; node.canvasProgress = ""; app.graph.setDirtyCanvas(true, true); }
}

async function generate(node) {
  if (busy) throw new Error("A canvas task is already running. Wait for it to finish.");
  busy = true;
  try { await runTask(node); } finally { busy = false; }
}

async function globalRefresh() {
  if (!armed) { app.extensionManager?.toast?.add({severity: "info", summary: "Canvas refresh is locked", detail: "Prepare refresh in Canvas Settings first.", life: 3500}); return; }
  if (busy) throw new Error("A canvas task is already running");
  const graph = snapshot();
  const data = await request("state", {graph});
  const jobs = data.plan.filter(p => ["changed", "upstream"].includes(p.status));
  if (data.plan.some(p => p.status === "blocked")) throw new Error("Canvas has missing inputs. Resolve blocked cards before global refresh.");
  if (!window.confirm(`Generate ${jobs.length} material tasks once? Model/prompt-only changes are not included.`)) return;
  armed = false;
  busy = true;
  try {
    for (const job of jobs) await runTask(app.graph.getNodeById(Number(job.id)) ?? app.graph.getNodeById(job.id), graph);
  } finally { busy = false; armed = false; }
}

async function importFile(node, file) {
  const settings = manager();
  if (!settings) throw new Error("Create Canvas Settings first");
  if (value(settings, "import_mode") !== "browser_upload") {
    throw new Error("Local-copy mode: browsers cannot expose the server path of a dropped file. Put it in the server input directory, enter local_path, then click Load / Refresh. No upload was started.");
  }
  const form = new FormData();
  form.append("canvas", JSON.stringify({graph: snapshot(), task: String(node.id)}));
  form.append("file", file);
  const response = await api.fetchApi("/turing/canvas/upload", {method: "POST", body: form});
  const asset = await response.json();
  if (!response.ok) throw new Error(asset.error);
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
    api.addEventListener("progress", event => {
      if (!activeTask || (event.detail.prompt_id && event.detail.prompt_id !== activeTask.promptId)) return;
      activeTask.node.canvasProgress = `${event.detail.value}/${event.detail.max}`;
      app.graph.setDirtyCanvas(true, true);
    });
    const queue = app.queuePrompt;
    app.queuePrompt = function (...args) {
      if (!manager() && !app.graph._nodes.some(isCard)) return queue.apply(this, args);
      return globalRefresh().catch(report);
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
    const changed = node.onWidgetChanged;
    node.onWidgetChanged = function(...args) { changed?.apply(this, args); scheduleRefresh(); };
    const connected = node.onConnectionsChange;
    node.onConnectionsChange = function(...args) { connected?.apply(this, args); scheduleRefresh(); };
    const removed = node.onRemoved;
    node.onRemoved = function(...args) { removed?.apply(this, args); this.canvasMedia?.pause?.(); scheduleRefresh(); };
    node.size = [390, 350];
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
      directory.mouse = () => { directoryPicker(node).catch(report); return true; };
      button(node, "选择工作目录", () => directoryPicker(node));
      button(node, "允许 / 取消全局执行一次", async () => {
        if (busy) throw new Error("Wait for the current canvas run before preparing another refresh");
        await refresh(); armed = !armed; app.graph.setDirtyCanvas(true, true);
      });
      button(node, "执行已变化的节点一次", globalRefresh);
      button(node, "Save Canvas Project", async () => request("save", {graph: snapshot(), workflow: app.graph.serialize()}));
      button(node, "Open Saved Project", async () => {
        if (!window.confirm("Replace the current canvas with the saved project?")) return;
        await app.loadGraphData(await request("load", {graph: snapshot()}));
      });
    } else if (["TuringCanvasImage", "TuringCanvasVideo", "TuringCanvasAudio"].includes(type)) {
      button(node, "选择项目素材", () => history(node));
      button(node, "导入素材", async () => {
        if (value(manager(), "import_mode") === "local_copy") {
          const asset = await request("import", {graph: snapshot(), task: String(node.id)});
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
        const widget = node.widgets.find(w => w.name === "model_prompt");
        const before = widget.value;
        const result = await request("enhance", {graph: snapshot(), task: String(node.id)});
        if (widget.value !== before && !window.confirm("Model prompt was edited during enhancement. Replace it?")) return;
        node.properties.canvasPreviousPrompt = widget.value;
        widget.value = result.model_prompt;
        app.graph.setDirtyCanvas(true, true);
      });
      node.widgets.splice(node.widgets.indexOf(promptButton), 1);
      node.widgets.splice(node.widgets.findIndex(w => w.name === "model_prompt"), 0, promptButton);
      button(node, "历史生成结果", () => history(node));
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
      const widget = node.addDOMWidget("material_preview", "canvas_preview", media, {serialize: false, hideOnZoom: true});
      widget.canvasPreview = true;
      widget.computeSize = () => [350, tag === "audio" ? 65 : 220];
      node.canvasMedia = media;
    }
    if (type === "TuringCanvasH3") button(node, "生成", () => generate(node));
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
    scheduleRefresh();
  },
  beforeConfigureGraph() { configuring = true; },
  afterConfigureGraph() { configuring = false; armed = false; filterTypes(); refresh().catch(() => {}); },
});
