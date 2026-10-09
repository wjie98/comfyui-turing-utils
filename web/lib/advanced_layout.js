// Classic frontend compatibility: drawing and layout must agree on advanced
// visibility. Opt in per node; never modify global prototypes or widget order.
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
