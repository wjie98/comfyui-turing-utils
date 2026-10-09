import { app } from "../../scripts/app.js";
import { migrateNoiseGrid } from "./lib/node_migrations.js";
import { advancedLast, isInternalNode, hideInternalNode } from "./lib/widget_layout.js";
import { stableInputRows, stableRowNames } from "./lib/stable_inputs.js";

app.registerExtension({
  name: "TuringUtils.NodeConfiguration",
  beforeRegisterNodeDef(type, data) {
    if (isInternalNode(data.name)) hideInternalNode(type);
  },
  setup() {
    const store = window.comfyAPI?.nodeDefStore?.useNodeDefStore?.()
      ?? app.extensionManager?._p?._s?.get("nodeDef");
    store?.registerNodeDefFilter?.({id: "turing.internal", predicate: def => !isInternalNode(def.name)});
  },
  nodeCreated(node) {
    if (!node.comfyClass?.startsWith("TuringUtils")) return;
    const isChat = node.comfyClass === "TuringUtilsMultimodalPromptChat";
    const rowNames = stableRowNames[node.comfyClass];
    if (!isChat && !rowNames) return;
    const restoreInputs = stableInputRows(node, rowNames ?? []);
    // The legacy canvas reads widget.advanced; Nodes 2.0 reads options.advanced.
    // Bridge the schema flag, keeping the frontend's own toggle and persistence.
    const sync = () => {
      restoreInputs();
      if (!isChat) return;
      for (const widget of node.widgets ?? []) {
        if (widget.options?.advanced !== undefined && widget.advanced !== widget.options.advanced) {
          widget.advanced = widget.options.advanced;
        }
      }
    };
    sync();
    const layoutWidgets = node.getLayoutWidgets;
    if (layoutWidgets && node.isWidgetVisible) {
      node.getLayoutWidgets = function (...args) {
        // DynamicCombo can create children after nodeCreated/onConfigure.
        sync();
        const widgets = layoutWidgets.apply(this, args);
        if (!isChat) return widgets;
        const visible = globalThis.LiteGraph?.vueNodesMode ? widgets : widgets.filter(w => this.isWidgetVisible(w));
        // Sort the display only: widgets_values must retain its serialized order.
        return advancedLast(visible);
      };
    }
    // Size new nodes compactly; onConfigure still restores users' saved sizes.
    if (isChat && node.hasAdvancedWidgets?.()) node.setSize(node.computeSize());
    for (const method of ["onConfigure", "onWidgetChanged", "onConnectionsChange"]) {
      const original = node[method];
      node[method] = function (...args) {
        const result = original?.apply(this, args);
        sync();
        queueMicrotask(sync);
        return result;
      };
    }
  },
  beforeConfigureGraph(graph) {
    migrateNoiseGrid(graph);
  },
});
