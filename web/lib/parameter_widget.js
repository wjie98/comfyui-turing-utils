import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
import { DYNAMIC_COMBO, dynamicParameterWidget } from "./dynamic_parameter.js";

// Use the same native controls on endpoint workflows and projected cards.
export function parameterWidget(node, name, type, value, callback, options = {}) {
  if (type === DYNAMIC_COMBO)
    return dynamicParameterWidget(node, name, value, callback, options);
  const widget =
    type === "STRING"
      ? ComfyWidgets.STRING(
          node,
          name,
          ["STRING", { default: value, multiline: options.multiline ?? true }],
          app,
        ).widget
      : node.addWidget(
          type === "BOOLEAN" ? "toggle" : type === "COMBO" ? "combo" : "number",
          name,
          value,
          () => {},
          {
            step: type === "INT" ? 10 : 1,
            precision: type === "INT" ? 0 : 3,
            ...options,
          },
        );
  widget.value = value;
  widget.callback = callback;
  widget.options = { ...widget.options, serialize: false };
  return widget;
}
