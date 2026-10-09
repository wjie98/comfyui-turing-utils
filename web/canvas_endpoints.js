import { app } from "../../scripts/app.js";
import { INPUTS, OUTPUTS, entries, syncPorts } from "./lib/canvas_ports.js";
import { installEndpoint } from "./lib/endpoint_interaction.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
  name: "TuringUtils.CanvasEndpoints",
  nodeCreated(node) {
    const type = node.comfyClass || node.constructor.type || node.type;
    if (type?.startsWith("TuringMaterial")) {
      const kind = type.slice("TuringMaterial".length).toLowerCase();
      const id = node.widgets?.find((w) => w.name === "stub_id");
      if (id) {
        if (!id.value) id.value = crypto.randomUUID();
      }
      if (kind !== "text") {
        const upload = async (file) => {
          const body = new FormData();
          body.append("image", file);
          const response = await api.fetchApi("/upload/image", {
            method: "POST",
            body,
          });
          if (!response.ok) throw Error("素材上传失败");
          const data = await response.json();
          node.widgets.find((w) => w.name === "file").value =
            (data.subfolder ? data.subfolder + "/" : "") + data.name;
          node.graph?.setDirtyCanvas(true, true);
        };
        const picker = () => {
          const input = document.createElement("input");
          input.type = "file";
          input.accept = kind + "/*";
          input.onchange = () =>
            upload(input.files[0]).catch((e) =>
              app.extensionManager.toast.add({
                severity: "error",
                summary: e.message,
              }),
            );
          input.click();
        };
        node.addWidget("button", "导入素材", null, picker, {
          serialize: false,
        });
        node.onDragOver = () => true;
        node.onDragDrop = (event) => {
          const file = event.dataTransfer?.files?.[0];
          if (!file) return false;
          upload(file).catch((e) =>
            app.extensionManager.toast.add({
              severity: "error",
              summary: e.message,
            }),
          );
          return true;
        };
      }
    }
    if (![INPUTS, OUTPUTS].includes(type)) return;
    installEndpoint(node);
    queueMicrotask(() => syncPorts(node, entries(node)));
  },
  loadedGraphNode(node) {
    if ([INPUTS, OUTPUTS].includes(node.type)) syncPorts(node, entries(node));
  },
});
