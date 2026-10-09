// Classic frontend compatibility: drawing and layout must agree on advanced
// visibility. Opt in per node; never modify global prototypes or widget order.
// TEMPORARY UPSTREAM WORKAROUND — reviewed 2026-10-09.
// Reproduced on frontend 1.53.6: options.advanced needs flag propagation, and
// getLayoutWidgets includes folded advanced widgets, leaving blank resize space.
// Layout omission also remains in v1.57.0 / main (source review, not runtime test):
// https://github.com/Comfy-Org/ComfyUI_frontend/blob/v1.57.0/src/lib/litegraph/src/LGraphNode.ts#L4217
// REMOVAL: during frontend upgrades or when an upstream fix is encountered,
// test WITHOUT this helper. Once native flag propagation and folded layout pass
// the classic Chat resize/toggle/DynamicCombo/reload regressions, proactively
// delete this module and web/chat_advanced.js; keep native regression coverage.
// Do not retain it as a permanent compatibility layer or trust version alone.
const installed = new WeakSet();

export function enableAdvancedLayout(node) {
  if (installed.has(node)) return;
  installed.add(node);
  const syncFlags = () => {
    for (const widget of node.widgets ?? []) {
      if (typeof widget.options?.advanced === "boolean")
        widget.advanced = widget.options.advanced;
    }
  };
  syncFlags();
  const layout = node.getLayoutWidgets;
  node.getLayoutWidgets = function (...args) {
    syncFlags(); // Includes newly created native DynamicCombo branch widgets.
    return layout.apply(this, args).filter(
      (widget) => !widget.advanced || this.showAdvanced,
    );
  };
}
