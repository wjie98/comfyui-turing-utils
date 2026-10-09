import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createH3Template } from "./lib/material_h3_template.js";
import {
  INPUTS,
  OUTPUTS,
  POSITION,
  addPort,
  entries,
  syncPorts,
} from "./lib/canvas_ports.js";

let editing = null;
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
export function bindMaterials(graph = app.graph) {
  let input = graph._nodes.find((n) => n.type === INPUTS),
    output = graph._nodes.find((n) => n.type === OUTPUTS);
  for (const [type, node] of [
    [INPUTS, input],
    [OUTPUTS, output],
  ])
    if (!node) {
      const n = LiteGraph.createNode(type);
      graph.add(n);
      if (type === INPUTS) input = n;
      else output = n;
    }
  for (const material of graph._nodes.filter((n) =>
    n.type?.startsWith("TuringMaterial"),
  )) {
    const identity = material.widgets.find((w) => w.name === "stub_id");
    if (identity && !identity.value) identity.value = crypto.randomUUID();
    const position = material.inputs.findIndex((i) => i.name === "position");
    if (position < 0) continue;
    if (!material.inputs[position].link) {
      const p = addPort(input, material.title, POSITION, "position");
      input.connect(p.slot, material, position);
    }
    const exported = (material.outputs[0].links || []).some(
      (id) => graph.links[id]?.target_id === output.id,
    );
    if (!exported) {
      const p = addPort(output, material.title, material.outputs[0].type);
      material.connect(
        0,
        output,
        output.inputs.findIndex((i) => i.name === `port_${p.slot}`),
      );
    }
  }
  for (const node of graph._nodes) {
    for (const name of node.properties.materialFields || []) {
      const widget = node.widgets?.find((w) => w.name === name);
      const slot = node.inputs?.findIndex((i) => i.name === name);
      if (!widget || slot < 0 || node.inputs[slot].link != null) continue;
      const type = node.inputs[slot].type;
      if (!["STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"].includes(type))
        continue;
      const port = addPort(input, `${node.title} · ${name}`, type);
      const ps = entries(input),
        entry = ps.find((p) => p.id === port.id);
      entry.default = widget.value;
      if (type === "COMBO") {
        const values = widget.options?.values;
        entry.options = typeof values === "function" ? values() : values;
      }
      syncPorts(input, ps);
      input.connect(port.slot, node, slot);
    }
  }
  const content = graph._nodes.filter((n) => n !== input && n !== output);
  const y = content.length ? Math.min(...content.map((n) => n.pos[1])) : 0;
  input.pos = [
    content.length ? Math.min(...content.map((n) => n.pos[0])) - 360 : 0,
    y,
  ];
  output.pos = [
    content.length
      ? Math.max(...content.map((n) => n.pos[0] + n.size[0])) + 100
      : 1200,
    y,
  ];
  graph.setDirtyCanvas(true, true);
}
async function saveTemplate() {
  const name = prompt("卡片模板文件名（不覆盖已有文件）", "我的卡片.json");
  if (!name) return;
  const { workflow, output } = await app.graphToPrompt();
  await request("template/save", { name, workflow, prompt: output });
  app.extensionManager.toast.add({
    severity: "success",
    summary: "已保存到卡片库",
    detail: name,
    life: 3000,
  });
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
      id: "Turing.MaterialWorkspace.Bind",
      label: "添加端点并绑定素材桩",
      function: () => {
        try {
          bindMaterials();
        } catch (e) {
          show(e);
        }
      },
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
      id: "Turing.MaterialWorkspace.H3",
      label: "添加 H3 卡片工作流",
      function: () => {
        try {
          createH3Template(app);
          bindMaterials();
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
        "Turing.MaterialWorkspace.Bind",
        "Turing.MaterialWorkspace.SaveTemplate",
        "Turing.MaterialWorkspace.SaveInstance",
        "Turing.MaterialWorkspace.H3",
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
              directory,
              asset: selected?.asset || "",
              start: selected?.start || 0,
              end: selected?.end || 0,
              include_audio: selected?.include_audio ?? true,
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
