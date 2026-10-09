import { app } from "../../scripts/app.js";
import {
  INPUTS,
  OUTPUTS,
  entries,
  syncPorts,
  editPorts,
} from "./lib/canvas_ports.js";

function hide(widget) {
  if (!widget) return;
  widget.type = "hidden";
  widget.computeSize = () => [0, -4];
}
app.registerExtension({
  name: "TuringUtils.CanvasEndpoints",
  nodeCreated(node) {
    const type = node.comfyClass || node.constructor.type || node.type;
    if (type?.startsWith("TuringMaterial")) {
      const id = node.widgets?.find((w) => w.name === "stub_id");
      if (id) {
        if (!id.value) id.value = crypto.randomUUID();
        hide(id);
      }
    }
    if (![INPUTS, OUTPUTS].includes(type)) return;
    hide(node.widgets.find((w) => w.name === "ports"));
    node.addWidget("button", "编辑端点与排序", null, () => editPorts(node), {
      serialize: false,
    });
    queueMicrotask(() => syncPorts(node, entries(node)));
  },
  loadedGraphNode(node) {
    if ([INPUTS, OUTPUTS].includes(node.type)) syncPorts(node, entries(node));
  },
});
