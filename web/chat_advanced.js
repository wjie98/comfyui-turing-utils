import { app } from "../../scripts/app.js";
import { enableAdvancedLayout } from "./lib/advanced_layout.js";
app.registerExtension({
  name: "TuringUtils.ChatAdvanced",
  nodeCreated(node) {
    if (node.comfyClass !== "TuringUtilsMultimodalPromptChat") return;
    enableAdvancedLayout(node);
  },
});
