import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { INPUTS, entries, syncPorts } from "./lib/canvas_ports.js";
import { DirectoryPicker } from "./lib/directory_picker.js";
import { request, report, action, prompt } from "./workspace/api.js";
import {
  button,
  field,
  url,
  controls,
  releasePlayer,
  clearPosters,
  stopActivePreview,
} from "./workspace/material_controls.js";
import {
  ROOT,
  CARD,
  jobs,
  project,
  editor,
  enqueueSave,
  saveProject,
  setProjectStatus,
} from "./workspace/project_state.js";

let templates = [];
export { request } from "./workspace/api.js";
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
const directoryPicker = (create) =>
  new DirectoryPicker(request, prompt, (message) =>
    app.extensionManager.dialog.confirm({ title: "删除空目录", message }),
  ).choose(create);
function configureCard(node, card, document, directory) {
  if (card.error) throw Error(card.error);
  node.properties.card = card;
  node.properties.directory = directory;
  node.properties.values = Object.fromEntries(
    card.fields.map((f) => [f.input, document.prompt[f.node].inputs[f.input]]),
  );
  const names = new Map(card.outputs.map((p) => [p.source, p.name]));
  node.properties.materials = card.materials.map((id) => ({
    id,
    name: names.get(id) || "素材",
    kind: document.prompt[id].class_type.replace("TuringMaterial", "").toLowerCase(),
    selection: document.selections[id] || {},
    executable: ["value", "images", "text"].some((name) =>
      Array.isArray(document.prompt[id].inputs[name]),
    ),
  }));
  node.inputs = card.ports.map((p) => ({ name: p.name, type: p.type, link: null }));
  node.outputs = card.outputs.map((p) => ({ name: p.name, type: p.type, links: [] }));
}
function prepare({ workflow, document, statistics }) {
  workflow.nodes.find((n) => n.type === ROOT).properties.statistics = statistics;
  const cards = new Map(document.cards.map((c) => [c.id, c]));
  const outputs = new Map();
  for (const node of workflow.nodes) {
    if (node.type !== CARD) continue;
    configureCard(
      node,
      cards.get(node.properties.instance),
      document,
      workflow.extra.turing_project.directory,
    );
    node.properties.card.outputs.forEach((p, slot) => {
      const key = p.source + ":" + p.source_slot;
      if (!outputs.has(key)) outputs.set(key, { node, slot });
    });
  }
  let id = 0;
  for (const target of workflow.nodes) {
    if (target.type !== CARD) continue;
    for (const [slot, port] of target.properties.card.ports.entries()) {
      const binding = target.properties.card.bindings?.[port.id];
      const source = binding && outputs.get(binding.source + ":" + binding.slot);
      if (!source) continue;
      workflow.links.push([
        ++id,
        source.node.id,
        source.slot,
        target.id,
        slot,
        port.type,
      ]);
      target.inputs[slot].link = id;
      source.node.outputs[source.slot].links.push(id);
    }
  }
  workflow.last_link_id = id;
  return workflow;
}
export async function openProject(directory, create = false) {
  templates = (await request("templates")).items;
  await openTab(
    prepare(await request(create ? "project/create" : "project/open", { directory })),
    `Canvas - ${directory}.json`,
  );
}
export { saveProject } from "./workspace/project_state.js";
export async function addCard(spec, position) {
  const state = { ...project() },
    directory = state.directory;
  await saveProject(directory);
  const result = await enqueueSave(state, async (revision) => {
    const added = await request("card/add", { directory, revision, ...spec });
    const definition = {
      type: CARD,
      title: added.document.cards[0].title,
      properties: { instance: added.id },
      pos: position || [360, 20],
      size: [320, 300],
    };
    configureCard(definition, added.document.cards[0], added.document, directory);
    if (project()?.directory === directory) {
      const node = LiteGraph.createNode(CARD);
      node.configure(definition);
      app.graph.add(node);
    } else {
      const tab = app.extensionManager.workflow.openWorkflows.find(
        (w) => w.activeState?.extra?.turing_project?.directory === directory,
      );
      if (tab) {
        definition.id = ++tab.activeState.last_node_id;
        tab.activeState.nodes.push(definition);
      }
    }
    setProjectStatus(directory, "已保存", added.statistics);
    return added;
  });
  await saveProject(directory);
  return result;
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
  const overrides = data.parameters || data.workflow.extra.turing_card.overrides || {};
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
          if ((w.name === "file" || w.name === "audio") && selection?.asset)
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
  const parameters = Object.fromEntries(
    Object.values(output)
      .filter((n) => n.class_type === INPUTS)
      .flatMap((n) =>
        JSON.parse(n.inputs.ports)
          .filter((p) => p.kind === "parameter")
          .map((p) => [p.id, p.default]),
      ),
  );
  const result = await request(editing.id ? "card/save" : "template/save", {
    ...editing,
    workflow,
    prompt: output,
    parameters,
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
  templates = (await request("templates")).items;
}
async function editTemplate() {
  const { items } = await request("templates");
  const name = await choose(items.map((name) => ({ label: name, value: name })));
  if (!name) return;
  const data = await request("template/open", { name });
  data.workflow.extra.turing_editor = { name, revision: data.revision };
  await openTab(data.workflow, name);
}
async function refreshMaterial(node, m, directory) {
  m.selection = await request("selection", { directory, node: m.id });
  const w = node.widgets.find((w) => w.name === m.id);
  if (w)
    w.value =
      m.kind === "text"
        ? m.selection.text || ""
        : m.selection.asset?.split("/").at(-1) || "";
  await w?.refreshMaterialList?.();
  node.graph?.setDirtyCanvas(true, true);
  const poster = node.posters?.get(m.id);
  if (poster && m.selection.asset) poster.src = url(directory, m.selection.asset, true);
}
async function select(node, m, content) {
  const previous = m.pending || Promise.resolve();
  const task = previous.then(async () => {
    const directory = node.properties.directory;
    await request("select", {
      directory,
      node: m.id,
      selection: content,
      revision: m.selection.revision || 0,
    });
    await refreshMaterial(node, m, directory);
    await saveProject(directory);
  });
  m.pending = task.catch(() => {});
  return task;
}
async function importMedia(node, m, file) {
  const directory = node.properties.directory,
    body = new FormData();
  body.append("file", file);
  let data;
  try {
    const r = await api.fetchApi(
      "/turing/workspace/import?" + new URLSearchParams({ directory, kind: m.kind }),
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
  const state = { ...project() },
    trims = Object.assign({}, ...app.graph._nodes.map((n) => n.trims || {}));
  await saveProject(state.directory);
  // Keep execution preparation in the same project queue as autosaves: a
  // completed background job must not advance the revision during compilation.
  const compiled = await enqueueSave(state, async (revision) => ({
    ...(await request("compile", { ...state, revision, target: m.id, trims })),
    revision,
  }));
  jobs.set(compiled.run_id, { node, m, directory: state.directory });
  setProjectStatus(state.directory, "执行中");
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
    setProjectStatus(state.directory, "执行失败");
    throw Error(JSON.stringify(await r.json()));
  }
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
    for (const [name, type, key] of [
      ["项目名称", "STRING", "name"],
      ["素材像素上限（MP）", "FLOAT", "max_megapixels"],
    ]) {
      const settings = this.properties.settings;
      const widget =
        this.widgets.find((w) => w.name === name) ||
        field(
          this,
          name,
          type,
          settings[key],
          async (value) => {
            const state = this.graph.extra.turing_project;
            await saveProject(this.properties.directory);
            await enqueueSave(state, async (revision) => {
              const next = { ...this.properties.settings, [key]: value };
              const result = await request("project/settings", {
                directory: state.directory,
                revision,
                settings: next,
              });
              this.properties.settings = next;
              return result;
            });
          },
          { min: 0.01, step: 10, precision: 2, multiline: false },
        );
      widget.value = settings[key];
    }
    for (const name of ["画布统计", "保存状态"]) {
      if (!this.widgets.some((w) => w.name === name)) {
        const widget = this.addWidget("text", name, "", () => {}, {
          serialize: false,
        });
        widget.disabled = true;
      }
    }
    setProjectStatus(
      this.properties.directory,
      "已保存",
      this.properties.statistics,
      this,
    );
    this.setSize(this.computeSize());
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
    this.releasePlayer();
    clearPosters(this);
    for (const w of [...(this.widgets || [])]) this.removeWidget(w);
    const card = this.properties.card;
    if (!card) return;
    for (const f of card.fields)
      field(
        this,
        f.input,
        f.type,
        this.properties.values[f.input],
        async (value) => {
          const state = this.graph.extra.turing_project;
          await enqueueSave(state, (revision) =>
            request("patch", {
              ...state,
              revision,
              changes: [{ type: "input", id: f.node, name: f.input, value }],
            }),
          );
          this.properties.values[f.input] = value;
        },
        f.options,
      ).label = f.label;
    for (const m of this.properties.materials)
      controls(this, m, { select, importMedia, run });
    button(this, "编辑工作流", () => editCard(this));
    this.setSize(this.computeSize());
  }
  releasePlayer() {
    releasePlayer(this);
  }
  onRemoved() {
    this.releasePlayer();
    for (const w of this.widgets || []) w.onRemove?.();
    clearPosters(this);
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
      label: "新建画布…",
      function: action(async () => {
        const d = await directoryPicker(true);
        if (d) await openProject(d, true);
      }),
    },
    {
      id: "Turing.MaterialWorkspace.Open",
      label: "打开画布…",
      function: action(async () => {
        const d = await directoryPicker(false);
        if (d) await openProject(d);
      }),
    },
    {
      id: "Turing.MaterialWorkspace.SaveProject",
      label: "保存画布",
      function: action(saveProject),
    },
    {
      id: "Turing.MaterialWorkspace.New",
      label: "新建卡片",
      function: action(async () => {
        const r = await api.fetchApi("/turing/workspace/new-template");
        await openTab(await r.json(), `新卡片-${crypto.randomUUID().slice(0, 6)}.json`);
      }),
    },
    {
      id: "Turing.MaterialWorkspace.Edit",
      label: "编辑卡片…",
      function: action(editTemplate),
    },
    {
      id: "Turing.MaterialWorkspace.SaveTemplate",
      label: "保存为卡片",
      function: action(saveTemplate),
    },
    {
      id: "Turing.MaterialWorkspace.SaveInstance",
      label: "保存回卡片",
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
      if (document.hidden) stopActivePreview();
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
          has_submenu: true,
          submenu: {
            options: [
              ...[
                ["图片", "image"],
                ["视频", "video"],
                ["音频", "audio"],
                ["文本", "text"],
              ].map(([label, kind]) => ({
                content: label,
                callback: action(() => addCard({ kind }, Array.from(this.graph_mouse))),
              })),
              null,
              ...templates.map((name) => ({
                content: name,
                callback: action(() => addCard({ name }, Array.from(this.graph_mouse))),
              })),
            ],
          },
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
            setProjectStatus(job.directory, "执行未完成");
            report(Error("执行未完成，已有素材保持不变"));
            return;
          }
          await refreshMaterial(job.node, job.m, job.directory);
          await saveProject(job.directory);
        }),
      );
  },
  beforeConfigureGraph() {
    for (const node of app.rootGraph?._nodes || []) node.releasePlayer?.();
  },
});
