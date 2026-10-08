// Display order is independent of persisted widgets_values and socket indices.
export function advancedLast(widgets) {
  return [...widgets].sort((a, b) => Number(!!a.advanced) - Number(!!b.advanced));
}

export function h3SettingsLayout(widgets) {
  const shifts = widgets.filter(w => ["shift_video", "shift_audio"].includes(w.name));
  const result = widgets.filter(w => !shifts.includes(w));
  const anchor = result.findLastIndex(w => w.canvasLoraRow || w.name === "loras");
  result.splice(anchor + 1, 0, ...shifts);
  return result;
}

export function isInternalNode(name) {
  return name?.startsWith("_TuringUtils") || name?.startsWith("_TuringCanvas") || name === "TuringUtilsStagePath";
}

export function hideInternalNode(type) {
  // ComfyUI rewrites skip_list when DevMode changes. Internal implementation
  // nodes are never palette entries, even in development mode.
  Object.defineProperty(type, "skip_list", {configurable: true, get: () => true, set: () => {}});
}
