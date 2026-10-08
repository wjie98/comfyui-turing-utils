// These JSON fields are both editable controls and named connection targets.
// Use ordinary STRING sockets, not widget-backed sockets: legacy frontend
// conversion hooks otherwise hide the text editor when a link is attached.
// Keep the editor separately, without recreating sockets or changing indices.
export function stableInputRows(node, names) {
  const originals = new Map();
  for (const widget of node.widgets ?? []) {
    if (!names.includes(widget.name)) continue;
    originals.set(widget, Object.fromEntries(
      ["type", "computeSize", "draw", "mouse", "serializeValue", "disabled"].map(key => [key, widget[key]])
    ));
  }
  return () => {
    for (const [widget, original] of originals) {
      const slot = node.inputs?.find(input => input.widget?.name === widget.name || input.name === widget.name);
      if (slot?.widget) {
        delete slot.widget;
        delete slot.pos; // Drop the old widget-row anchor; lay out as a normal input.
      }
      const linked = slot?.link != null;
      if (widget.hidden || String(widget.type).startsWith("converted-widget")) {
        for (const key of ["type", "computeSize", "draw", "mouse", "serializeValue"]) {
          if (original[key] === undefined) delete widget[key];
          else widget[key] = original[key];
        }
      }
      widget.hidden = false;
      if (widget.options) widget.options.hidden = false;
      widget.disabled = linked || !!original.disabled;
    }
  };
}
