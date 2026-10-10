import { app } from "../../scripts/app.js";
import { INPUTS, OUTPUTS, entries, syncPorts } from "./lib/canvas_ports.js";
import { installEndpoint } from "./lib/endpoint_interaction.js";
import { expandDynamicParameters } from "./lib/dynamic_parameter.js";

app.registerExtension({
  name: "TuringUtils.CanvasEndpoints",
  setup() {
    const graphToPrompt = app.graphToPrompt;
    app.graphToPrompt = async function (...args) {
      const result = await graphToPrompt.apply(this, args);
      expandDynamicParameters(result.output);
      return result;
    };
  },
  beforeRegisterNodeDef(nodeType, data) {
    if (data.name !== "TuringMaterialAudio") return;
    // Native audio upload expects audioUI to exist during node construction,
    // before nodeCreated is called (the same order as Comfy's LoadAudio).
    const required = data.input.required;
    delete required.upload;
    required.audioUI = ["AUDIO_UI", {}];
    required.upload = ["AUDIOUPLOAD", {}];
  },
  nodeCreated(node) {
    const type = node.comfyClass || node.constructor.type || node.type;
    if (type?.startsWith("TuringMaterial")) {
      const kind = type.slice("TuringMaterial".length).toLowerCase();
      const id = node.widgets?.find((w) => w.name === "stub_id");
      if (id) {
        if (!id.value) id.value = crypto.randomUUID();
      }
      if (kind === "text" && !node.inputs.some((s) => s.name === "text"))
        node.addInput("text", "STRING");
      if (kind !== "text") {
        const file = node.widgets.find(
          (w) => w.name === (kind === "audio" ? "audio" : "file"),
        );
        if (file.value === "Loading...") file.value = "";
        if (kind === "audio") {
          node.widgets.find((w) => w.name === "audioUI").element.preload = "none";
        }
        for (const widget of node.widgets.filter((w) => w.name === "upload"))
          widget.label = "choose file to upload";
      }
      const executed = node.onExecuted;
      node.onExecuted = function (data) {
        executed?.call(this, data);
        const name = kind === "text" ? "text" : kind === "audio" ? "audio" : "file";
        const value = data.material_text?.[0] ?? data.material_file?.[0];
        const widget = this.widgets.find((w) => w.name === name);
        if (value !== undefined && widget) {
          widget.value = value;
          if (
            kind !== "text" &&
            Array.isArray(widget.options.values) &&
            !widget.options.values.includes(value)
          )
            widget.options.values.push(value);
          this.graph?.setDirtyCanvas(true, true);
        }
      };
    }
    if (![INPUTS, OUTPUTS].includes(type)) return;
    installEndpoint(node);
    queueMicrotask(() => syncPorts(node, entries(node)));
  },
  loadedGraphNode(node) {
    if ([INPUTS, OUTPUTS].includes(node.type)) syncPorts(node, entries(node));
  },
});
