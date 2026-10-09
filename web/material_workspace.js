import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { INPUTS, entries, syncPorts } from "./lib/canvas_ports.js";

const ROOT = "TuringCanvasProject",
  CARD = "TuringCanvasCard",
  jobs = new Map();
let playing = null;
const project = () => app.graph.extra?.turing_project;
const editor = () => app.graph.extra?.turing_editor;
export async function request(path, body = {}) {
  const r = await api.fetchApi(`/turing/workspace/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) throw Error(data.error || JSON.stringify(data));
  return data;
}
const report = (e) =>
  app.extensionManager.toast.add({
    severity: "error",
    summary: "Canvas",
    detail: e.message || String(e),
    life: 7000,
  });
const action =
  (fn) =>
  (...args) =>
    Promise.resolve()
      .then(() => fn(...args))
      .catch(report);
const prompt = (title, value = "") =>
  app.extensionManager.dialog.prompt({
    title,
    message: title,
    defaultValue: value,
  });
export async function openTab(workflow, name) {
  const p = workflow.extra?.turing_project,
    e = workflow.extra?.turing_editor;
  const existing = app.extensionManager.workflow.openWorkflows.find((w) => {
    const extra = w.activeState?.extra;
    return p
      ? extra?.turing_project?.directory === p.directory
      : e?.id
        ? extra?.turing_editor?.id === e.id &&
          extra.turing_editor.directory === e.directory
        : false;
  });
  await app.loadGraphData(workflow, true, true, existing || name);
}
function choose(items) {
  return new Promise((resolve) => {
    const menu = new LiteGraph.ContextMenu(
      items.map((i) => ({
        content: i.label,
        callback: () => resolve(i.value),
      })),
      {
        title: "Canvas",
        event: new MouseEvent("click", {
          clientX: window.innerWidth / 2,
          clientY: 160,
        }),
      },
    );
    const close = menu.close;
    menu.close = function (...args) {
      resolve(null);
      return close.apply(this, args);
    };
  });
}
async function directoryPicker(create) {
  let path = "";
  for (;;) {
    const { items } = await request("projects", { path });
    const picked = await choose([
      { label: `选择当前：output/${path}`, value: "select" },
      ...(path ? [{ label: "上一级", value: "parent" }] : []),
      ...items.map((p) => ({
        label: `${p.project ? "[Canvas] " : ""}${p.name}/`,
        value: p,
      })),
      ...(create ? [{ label: "新建文件夹…", value: "mkdir" }] : []),
    ]);
    if (picked === null) return null;
    if (picked === "select") {
      if (!path) throw Error("请选择 output 内的项目文件夹");
      return path;
    }
    if (picked === "parent") path = path.split("/").slice(0, -1).join("/");
    else if (picked === "mkdir") {
      const name = await prompt("新文件夹名称");
      if (!name) continue;
      if (/[\\/]/.test(name) || [".", ".."].includes(name))
        throw Error("请输入单个文件夹名称");
      const directory = [path, name].filter(Boolean).join("/");
      await request("directory/create", { directory });
      path = directory;
    } else path = picked.path;
  }
}
function prepare({ workflow, document }) {
  for (const node of workflow.nodes) {
    if (node.type !== CARD) continue;
    const card = document.cards.find((c) => c.id === node.properties.instance);
    if (card.error) throw Error(card.error);
    node.properties.card = card;
    node.properties.directory = workflow.extra.turing_project.directory;
    node.properties.values = Object.fromEntries(
      card.fields.map((f) => [
        f.input,
        document.prompt[f.node].inputs[f.input],
      ]),
    );
    node.properties.materials = card.materials.map((id) => ({
      id,
      name: card.outputs.find((p) => p.source === id)?.name || "素材",
      kind: document.prompt[id].class_type
        .replace("TuringMaterial", "")
        .toLowerCase(),
      selection: document.selections[id] || {},
      executable: !!(
        document.prompt[id].inputs.value || document.prompt[id].inputs.images
      ),
    }));
    node.inputs = card.ports.map((p) => ({
      name: p.name,
      type: p.type,
      link: null,
    }));
    node.outputs = card.outputs.map((p) => ({
      name: p.name,
      type: p.type,
      links: [],
    }));
  }
  let id = 0;
  for (const target of workflow.nodes.filter((n) => n.type === CARD))
    for (const [slot, port] of target.properties.card.ports.entries()) {
      const binding = target.properties.card.bindings?.[port.id];
      if (!binding) continue;
      const source = workflow.nodes.find(
        (n) =>
          n.type === CARD &&
          n.properties.card.outputs.some(
            (p) =>
              p.source === binding.source && p.source_slot === binding.slot,
          ),
      );
      if (!source) continue;
      const out = source.properties.card.outputs.findIndex(
        (p) => p.source === binding.source && p.source_slot === binding.slot,
      );
      workflow.links.push([++id, source.id, out, target.id, slot, port.type]);
      target.inputs[slot].link = id;
      source.outputs[out].links.push(id);
    }
  workflow.last_link_id = id;
  return workflow;
}
export async function openProject(directory, create = false) {
  await openTab(
    prepare(
      await request(create ? "project/create" : "project/open", { directory }),
    ),
    `Canvas - ${directory}.json`,
  );
}
export async function saveProject() {
  const state = project();
  if (!state) throw Error("当前不是素材画布");
  const workflow = app.graph.serialize();
  workflow.nodes = workflow.nodes.map((n) => ({
    id: n.id,
    type: n.type,
    pos: n.pos,
    size: n.size,
    properties: { instance: n.properties.instance },
  }));
  const result = await request("project/save", {
    ...state,
    workflow,
  });
  state.revision = result.revision;
}
async function addCard(spec) {
  await saveProject();
  const directory = project().directory;
  await request("card/add", { directory, ...spec });
  await openProject(directory);
}
async function editCard(node) {
  await saveProject();
  const directory = project().directory,
    id = node.properties.instance;
  const data = await request("card/open", { directory, id });
  data.workflow.extra.turing_editor = {
    directory,
    id,
    revision: data.revision,
  };
  await openTab(data.workflow, `${id}.json`);
  const overrides = data.workflow.extra.turing_card.overrides || {};
  const restore = (graph) => {
    for (const n of graph._nodes) {
      if (n.type === INPUTS) {
        const ports = entries(n);
        for (const p of ports)
          if (p.id in overrides) {
            p.default = overrides[p.id];
            const slot = n.inputs.findIndex((s) => s.name === `port_${p.slot}`);
            if (slot >= 0) n.disconnectInput(slot);
          }
        syncPorts(n, ports);
      }
      if (n.type?.startsWith("TuringMaterial")) {
        const stub = n.widgets.find((w) => w.name === "stub_id")?.value,
          selection = data.selections[`${id}:${stub}`];
        for (const w of n.widgets) {
          if (w.name === "text" && selection && "text" in selection)
            w.value = selection.text;
          if (w.name === "file" && selection?.asset)
            w.value = `${directory}/${selection.asset} [output]`;
        }
      }
      if (n.subgraph) restore(n.subgraph);
    }
  };
  restore(app.graph);
}
async function saveCard() {
  const editing = editor();
  if (!editing) throw Error("请先打开卡片工作流");
  const { workflow, output } = await app.graphToPrompt();
  delete workflow.extra.turing_editor;
  const result = await request(editing.id ? "card/save" : "template/save", {
    ...editing,
    workflow,
    prompt: output,
  });
  editing.revision = result.revision;
}
async function saveTemplate() {
  if (project()) throw Error("请在卡片工作流中保存模板");
  const name = await prompt("卡片文件名", editor()?.name || "我的卡片.json");
  if (!name) return;
  const { workflow, output } = await app.graphToPrompt();
  delete workflow.extra?.turing_editor;
  app.graph.extra.turing_editor = await request("template/save", {
    name,
    workflow,
    prompt: output,
    revision: editor()?.name === name ? editor().revision : undefined,
  });
}
async function editTemplate() {
  const { items } = await request("templates");
  const name = await choose(
    items.map((name) => ({ label: name, value: name })),
  );
  if (!name) return;
  const data = await request("template/open", { name });
  data.workflow.extra.turing_editor = { name, revision: data.revision };
  await openTab(data.workflow, name);
}
function button(node, name, fn) {
  return node.addWidget("button", name, null, action(fn), { serialize: false });
}
function field(node, name, type, value, fn, options = {}) {
  const w =
    type === "STRING"
      ? ComfyWidgets.STRING(
          node,
          name,
          ["STRING", { default: value, multiline: true }],
          app,
        ).widget
      : node.addWidget(
          type === "BOOLEAN" ? "toggle" : type === "COMBO" ? "combo" : "number",
          name,
          value,
          () => {},
          options,
        );
  w.value = value;
  w.callback = action(fn);
  w.options = { ...w.options, serialize: false };
  return w;
}
function url(directory, asset, thumbnail = false) {
  return api.apiURL(
    "/turing/workspace/asset?" +
      new URLSearchParams({
        directory,
        asset,
        ...(thumbnail ? { thumbnail: "1" } : {}),
      }),
  );
}
async function refreshMaterial(node, m, directory) {
  m.selection = await request("selection", { directory, node: m.id });
  const w = node.widgets.find((w) => w.name === m.id);
  if (w)
    w.value =
      m.kind === "text" ? m.selection.text || "" : m.selection.asset || "";
  node.graph?.setDirtyCanvas(true, true);
  const poster = node.posters?.get(m.id);
  if (poster && m.selection.asset)
    poster.src = url(directory, m.selection.asset, true);
}
async function select(node, m, content) {
  const directory = node.properties.directory;
  await request("select", {
    directory,
    node: m.id,
    selection: content,
    revision: m.selection.revision || 0,
  });
  await refreshMaterial(node, m, directory);
}
async function importMedia(node, m, file) {
  const directory = node.properties.directory,
    body = new FormData();
  body.append("file", file);
  let data;
  try {
    const r = await api.fetchApi(
      "/turing/workspace/import?" +
        new URLSearchParams({ directory, kind: m.kind }),
      { method: "POST", body },
    );
    data = await r.json();
    if (!r.ok) throw Error(data.error);
  } catch (error) {
    const path = await prompt(
      `上传失败：${error.message}。如文件已在服务器 input 中，可输入相对路径拷贝`,
    );
    if (!path) throw error;
    data = await request("copy", { directory, kind: m.kind, path });
  }
  await select(node, m, { asset: data.asset });
}
async function run(node, m) {
  await saveProject();
  const state = { ...project() },
    trims = Object.assign({}, ...app.graph._nodes.map((n) => n.trims || {})),
    compiled = await request("compile", { ...state, target: m.id, trims });
  jobs.set(compiled.run_id, { node, m, directory: state.directory });
  const r = await api.fetchApi("/prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt: compiled.prompt,
      prompt_id: compiled.run_id,
      client_id: api.clientId,
      partial_execution_targets: [m.id],
    }),
  });
  if (!r.ok) {
    jobs.delete(compiled.run_id);
    throw Error(JSON.stringify(await r.json()));
  }
}
function controls(node, m) {
  if (m.kind === "text")
    field(node, m.id, "STRING", m.selection.text || "", (value) =>
      select(node, m, { text: value }),
    ).label = m.name;
  else {
    if (["image", "video"].includes(m.kind)) {
      const img = new Image();
      img.loading = "lazy";
      img.style.cssText =
        "width:100%;height:100%;max-height:180px;object-fit:contain";
      if (m.selection.asset)
        img.src = url(node.properties.directory, m.selection.asset, true);
      node.posters ||= new Map();
      node.posters.set(m.id, img);
      const preview = node.addDOMWidget(`poster:${m.id}`, "image", img, {
        serialize: false,
      });
      preview.computeSize = () => [280, 180];
    }
    const w = field(
      node,
      m.id,
      "COMBO",
      m.selection.asset || "",
      (value) => select(node, m, { asset: value }),
      { values: [m.selection.asset || ""] },
    );
    w.label = m.name;
    button(node, `${m.kind} · 历史素材`, async () => {
      const data = await request("history", {
        directory: project().directory,
        kind: m.kind,
      });
      w.options.values = data.items.map((i) => i.id);
    });
    button(node, `${m.kind} · 导入`, () => {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = m.kind + "/*";
      input.onchange = action(async () => {
        if (!input.files[0]) return;
        await importMedia(node, m, input.files[0]);
      });
      input.click();
    });
    button(node, `${m.kind} · 预览`, () => {
      if (!m.selection.asset) return;
      playing?.releasePlayer();
      node.releasePlayer();
      const media = document.createElement(m.kind === "image" ? "img" : m.kind);
      media.style.cssText =
        "width:100%;height:100%;max-height:240px;object-fit:contain";
      if (m.kind !== "image") {
        media.controls = true;
        media.preload = "none";
        media.onloadedmetadata = () => {
          media.currentTime = node.trims[m.id]?.start || 0;
        };
        media.ontimeupdate = () => {
          const trim = node.trims[m.id] || {};
          if (trim.end && media.currentTime >= trim.end) {
            media.pause();
            media.currentTime = trim.start || 0;
          }
        };
      }
      media.src = url(
        node.properties.directory,
        m.selection.asset,
        m.kind === "image",
      );
      node.player = media;
      playing = node;
      node.playerWidget = node.addDOMWidget("preview", m.kind, media, {
        serialize: false,
      });
      node.playerWidget.computeSize = () => [
        280,
        m.kind === "audio" ? 54 : 180,
      ];
      node.setSize(node.computeSize());
      node.visibility = new IntersectionObserver((items) => {
        if (!items[0].isIntersecting) node.releasePlayer();
      });
      node.visibility.observe(media);
    });
  }
  if (["video", "audio"].includes(m.kind)) {
    node.trims ||= {};
    node.trims[m.id] = { start: 0, end: 0 };
    for (const [key, label] of [
      ["start", "起点（秒）"],
      ["end", "终点（秒，0 为末尾）"],
    ])
      field(
        node,
        `${m.name} · ${label}`,
        "FLOAT",
        0,
        (value) => {
          node.trims[m.id][key] = value;
        },
        { min: 0, step: 1, precision: 2 },
      );
  }
  if (m.executable) button(node, `${m.kind} · 执行到这里`, () => run(node, m));
}
class ProjectNode extends LiteGraph.LGraphNode {
  constructor() {
    super("Canvas Project");
    this.flags = { pinned: true };
    button(this, "保存项目", saveProject);
  }
  onConfigure() {
    this.flags.pinned = true;
    let w = this.widgets.find((w) => w.name === "项目目录");
    if (!w)
      w = this.addWidget("text", "项目目录", "", () => {}, {
        serialize: false,
      });
    w.value = this.properties.directory;
    w.disabled = true;
  }
}
class CardNode extends LiteGraph.LGraphNode {
  constructor() {
    super("Canvas Card");
  }
  onDragOver() {
    return true;
  }
  onDragDrop(event) {
    const file = event.dataTransfer?.files?.[0];
    if (!file) return false;
    const material = this.properties.materials.find((m) =>
      file.type.startsWith(m.kind + "/"),
    );
    if (!material) {
      report(Error("请将素材拖到对应类型的素材节点"));
      return true;
    }
    action(() => importMedia(this, material, file))();
    return true;
  }
  onConfigure() {
    for (const w of this.widgets || []) w.onRemove?.();
    this.widgets = [];
    const card = this.properties.card;
    if (!card) return;
    for (const f of card.fields)
      field(
        this,
        f.label,
        f.type,
        this.properties.values[f.input],
        async (value) => {
          const state = project(),
            result = await request("patch", {
              ...state,
              changes: [{ type: "input", id: f.node, name: f.input, value }],
            });
          state.revision = result.revision;
          this.properties.values[f.input] = value;
        },
        {
          values: f.options,
          step: f.type === "INT" ? 10 : 1,
          precision: f.type === "INT" ? 0 : 3,
        },
      );
    for (const m of this.properties.materials) controls(this, m);
    button(this, "编辑工作流", () => editCard(this));
    this.setSize(this.computeSize());
  }
  releasePlayer() {
    this.visibility?.disconnect();
    if (this.player) {
      this.player.pause?.();
      this.player.removeAttribute("src");
      this.player.load?.();
      this.player.remove();
      const i = this.widgets.indexOf(this.playerWidget);
      if (i >= 0) this.widgets.splice(i, 1);
      this.playerWidget.onRemove?.();
      this.player = null;
    }
    if (playing === this) playing = null;
  }
  onRemoved() {
    this.releasePlayer();
  }
}
app.registerExtension({
  name: "TuringUtils.MaterialWorkspace",
  registerCustomNodes() {
    LiteGraph.registerNodeType(ROOT, ProjectNode);
    LiteGraph.registerNodeType(CARD, CardNode);
  },
  commands: [
    {
      id: "Turing.MaterialWorkspace.Create",
      label: "新建素材画布…",
      function: action(async () => {
        const d = await directoryPicker(true);
        if (d) await openProject(d, true);
      }),
    },
    {
      id: "Turing.MaterialWorkspace.Open",
      label: "打开素材画布…",
      function: action(async () => {
        const d = await directoryPicker(false);
        if (d) await openProject(d);
      }),
    },
    {
      id: "Turing.MaterialWorkspace.SaveProject",
      label: "保存素材画布",
      function: action(saveProject),
    },
    {
      id: "Turing.MaterialWorkspace.New",
      label: "新建 Canvas 卡片",
      function: action(async () => {
        const r = await api.fetchApi("/turing/workspace/new-template");
        await openTab(
          await r.json(),
          `新卡片-${crypto.randomUUID().slice(0, 6)}.json`,
        );
      }),
    },
    {
      id: "Turing.MaterialWorkspace.Edit",
      label: "编辑 Canvas 卡片…",
      function: action(editTemplate),
    },
    {
      id: "Turing.MaterialWorkspace.SaveTemplate",
      label: "保存为 Canvas 卡片",
      function: action(saveTemplate),
    },
    {
      id: "Turing.MaterialWorkspace.SaveInstance",
      label: "保存回 Canvas 卡片",
      function: action(saveCard),
    },
  ],
  menuCommands: [
    {
      path: ["Turing Utils"],
      commands: [
        "Create",
        "Open",
        "SaveProject",
        "New",
        "Edit",
        "SaveTemplate",
        "SaveInstance",
      ].map((n) => "Turing.MaterialWorkspace." + n),
    },
  ],
  setup() {
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) playing?.releasePlayer();
    });
    const queue = app.queuePrompt;
    app.queuePrompt = function (...args) {
      if (project()) {
        report(Error("请使用素材的执行按钮，画布不执行整图"));
        return Promise.resolve();
      }
      return queue.apply(this, args);
    };
    const menu = app.canvas.getCanvasMenuOptions;
    app.canvas.getCanvasMenuOptions = function (...args) {
      if (!project()) return menu.apply(this, args);
      const defaults = menu.apply(this, args);
      return [
        {
          content: "添加节点…",
          callback: action(async () => {
            const { items } = await request("templates");
            const spec = await choose([
              ...[
                ["图片", "image"],
                ["视频", "video"],
                ["音频", "audio"],
                ["文本", "text"],
              ].map(([label, kind]) => ({ label, value: { kind } })),
              ...items.map((name) => ({ label: name, value: { name } })),
            ]);
            if (spec) await addCard(spec);
          }),
        },
        { content: "保存项目", callback: action(saveProject) },
        ...defaults.slice(1),
      ];
    };
    for (const event of [
      "execution_success",
      "execution_error",
      "execution_interrupted",
    ])
      api.addEventListener(
        event,
        action(async (e) => {
          const job = jobs.get(e.detail.prompt_id);
          if (!job) return;
          jobs.delete(e.detail.prompt_id);
          if (event !== "execution_success") {
            report(Error("执行未完成，已有素材保持不变"));
            return;
          }
          await refreshMaterial(job.node, job.m, job.directory);
        }),
      );
  },
  beforeConfigureGraph() {
    for (const node of app.rootGraph?._nodes || []) node.releasePlayer?.();
  },
});
