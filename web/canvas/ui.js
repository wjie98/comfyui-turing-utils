import { app } from "../../../scripts/app.js";

export const clamp = (value, min, max) => Math.min(max, Math.max(min,
  Number.isFinite(Number(value)) ? Number(value) : min));

export function choiceMenu(values, event, title, callback) {
  return new LiteGraph.ContextMenu(values, {event, title, className: "dark",
    scale: Math.max(1, app.canvas.ds.scale), callback});
}

// A response may arrive after a newer request, node removal, project switch or reload.
export function requestGuard(node, context = () => null) {
  let revision = 0;
  return () => {
    const token = ++revision, graph = app.graph, key = context();
    return () => token === revision && graph === app.graph && graph.getNodeById(node.id) === node && context() === key;
  };
}

export function boundedPreview(node, box, audio) {
  const minHeight = audio ? 95 : 180, maxHeight = 900;
  const widget = node.addDOMWidget("material_preview", "canvas_preview", box, {
    serialize: false, hideOnZoom: true, getMinHeight: () => minHeight, getMaxHeight: () => maxHeight,
  });
  widget.canvasPreview = true;
  // Never feed the current width/height into the layout minimum (resize feedback).
  widget.computeSize = () => [300, minHeight];
  const resize = node.onResize;
  node.onResize = function(size) {
    resize?.call(this, size);
    const minimum = this.computeSize();
    size[0] = clamp(size[0], Math.max(320, Math.min(1600, minimum[0])), 1600);
    size[1] = clamp(size[1], minimum[1], minimum[1] + maxHeight - minHeight);
    this.properties.canvasPreviewHeight = clamp(size[1] - minimum[1] + minHeight, minHeight, maxHeight);
  };
  const configure = node.onConfigure;
  node.onConfigure = function(...args) {
    const result = configure?.apply(this, args);
    this.onResize(this.size);
    return result;
  };
  return widget;
}
