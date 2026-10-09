import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { INPUTS, entries, syncPorts } from "./lib/canvas_ports.js";

let editing = null;
let templateEditing = null;
const channel = new BroadcastChannel("turing-material-workspace");
async function request(path, body) {
  const response = await api.fetchApi(`/turing/workspace/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw Error(data.error || JSON.stringify(data));
  return data;
}
function show(error) {
  app.extensionManager.toast.add({
    severity: "error",
    summary: "素材卡片",
    detail: error.message || String(error),
    life: 7000,
  });
}
function openWorkspace() {
  const url = new URL(api.apiURL("/turing/workspace/"), location.href);
  if (editing) url.searchParams.set("directory", editing.directory);
  if (api.user) url.searchParams.set("user", api.user);
  window.open(url, "_blank", "noopener");
}
async function saveTemplate() {
  const name = prompt("卡片文件名", templateEditing?.name || "我的卡片.json");
  if (!name) return;
  const replacing = templateEditing?.name === name;
  if (replacing && !confirm("更新卡片库中的这张卡片？已有画布实例不受影响。"))
    return;
  const { workflow, output } = await app.graphToPrompt();
  templateEditing = await request("template/save", {
    name,
    workflow,
    prompt: output,
    revision: replacing ? templateEditing.revision : undefined,
  });
  app.extensionManager.toast.add({
    severity: "success",
    summary: "已保存到卡片库",
    detail: name,
    life: 3000,
  });
}
async function editTemplate() {
  const { items } = await request("templates", {});
  const dialog = document.createElement("dialog");
  const select = document.createElement("select");
  select.append(...items.map((name) => new Option(name, name)));
  const open = document.createElement("button");
  open.textContent = "打开卡片";
  open.disabled = !items.length;
  open.onclick = async () => {
    if (!confirm("打开卡片会替换当前编辑器内容，请先保存。继续？")) return;
    try {
      const data = await request("template/open", { name: select.value });
      await app.loadGraphData(data.workflow);
      templateEditing = { name: select.value, revision: data.revision };
      editing = null;
      dialog.close();
    } catch (e) {
      show(e);
    }
  };
  const close = document.createElement("button");
  close.textContent = "取消";
  close.onclick = () => dialog.close();
  dialog.append(select, open, close);
  dialog.onclose = () => dialog.remove();
  document.body.append(dialog);
  dialog.showModal();
}
async function saveInstance() {
  if (!editing) throw Error("请从画布卡片的“编辑工作流”进入");
  const { workflow, output } = await app.graphToPrompt();
  const result = await request("card/save", {
    ...editing,
    workflow,
    prompt: output,
  });
  editing.revision = result.revision;
  channel.postMessage({ directory: editing.directory });
  app.extensionManager.toast.add({
    severity: "success",
    summary: "已保存回卡片",
    detail: editing.id,
    life: 3000,
  });
}
app.registerExtension({
  name: "TuringUtils.MaterialWorkspace",
  commands: [
    {
      id: "Turing.MaterialWorkspace.Open",
      label: "打开素材画布",
      function: openWorkspace,
    },
    {
      id: "Turing.MaterialWorkspace.Edit",
      label: "编辑 Canvas 卡片…",
      function: () => editTemplate().catch(show),
    },
    {
      id: "Turing.MaterialWorkspace.SaveTemplate",
      label: "保存为 Canvas 卡片",
      function: () => saveTemplate().catch(show),
    },
    {
      id: "Turing.MaterialWorkspace.SaveInstance",
      label: "保存回 Canvas 卡片",
      function: () => saveInstance().catch(show),
    },
    {
      id: "Turing.MaterialWorkspace.New",
      label: "新建 Canvas 卡片",
      function: async () => {
        if (!confirm("新建卡片会替换当前编辑器内容，请先保存工作流。继续？"))
          return;
        try {
          const response = await api.fetchApi("/turing/workspace/new-template");
          if (!response.ok) throw Error("无法加载基础卡片");
          await app.loadGraphData(await response.json());
          editing = null;
          templateEditing = null;
        } catch (e) {
          show(e);
        }
      },
    },
  ],
  menuCommands: [
    {
      path: ["Turing Utils"],
      commands: [
        "Turing.MaterialWorkspace.Open",
        "Turing.MaterialWorkspace.New",
        "Turing.MaterialWorkspace.Edit",
        "Turing.MaterialWorkspace.SaveTemplate",
        "Turing.MaterialWorkspace.SaveInstance",
      ],
    },
  ],
  async setup() {
    const params = new URL(location.href).searchParams,
      directory = params.get("turing_workspace"),
      id = params.get("card");
    if (!directory || !id) return;
    if (!confirm("打开此卡片的工作流副本？请先保存当前标签页的编辑。")) return;
    try {
      const data = await request("card/open", { directory, id });
      await app.loadGraphData(data.workflow);
      const overrides = data.workflow.extra?.turing_card?.overrides || {};
      const walk = (graph) => {
        for (const node of graph._nodes) {
          if (node.type === INPUTS) {
            const ps = entries(node);
            for (const p of ps)
              if (p.id in overrides) {
                p.default = overrides[p.id];
                const slot = node.inputs.findIndex(
                  (i) => i.name === `port_${p.slot}`,
                );
                if (slot >= 0) node.disconnectInput(slot);
              }
            syncPorts(node, ps);
          }
          if (node.type?.startsWith("TuringMaterial")) {
            const stub = node.widgets.find((w) => w.name === "stub_id")?.value;
            const selected = data.selections?.[`${id}:${stub}`];
            const values = {
              ...(selected && "text" in selected
                ? { text: selected.text }
                : {}),
              ...(selected?.asset
                ? { file: `${directory}/${selected.asset} [output]` }
                : {}),
            };
            for (const widget of node.widgets)
              if (widget.name in values) widget.value = values[widget.name];
          }
          if (node.subgraph) walk(node.subgraph);
        }
      };
      walk(app.graph);
      editing = { directory, id, revision: data.revision };
      app.extensionManager.toast.add({
        severity: "info",
        summary: "正在编辑卡片副本",
        detail: `${directory}/cards/${id}/workflow.json；完成后使用“保存回 Canvas 卡片”`,
        life: 10000,
      });
    } catch (error) {
      show(error);
    }
  },
});
