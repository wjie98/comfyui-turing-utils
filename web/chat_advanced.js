import { app } from "../../scripts/app.js";

// Frontend 1.53.6 supplies options.advanced, but classic LiteGraph reads
// widget.advanced. Layout, socket positions and disabled state remain native.
app.registerExtension({
  name: "TuringUtils.ChatAdvanced",
  nodeCreated(node) {
    if (node.comfyClass !== "TuringUtilsMultimodalPromptChat") return;
    const layout = node.getLayoutWidgets;
    node.getLayoutWidgets = function (...args) {
      for (const widget of this.widgets ?? [])
        widget.advanced = !!widget.options?.advanced;
      return layout.apply(this, args);
    };
  },
});
