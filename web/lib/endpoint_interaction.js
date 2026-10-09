import {
  POSITION,
  INPUTS,
  MATERIAL_TYPES,
  entries,
  syncPorts,
} from "./canvas_ports.js";
import { PortReorder } from "./port_reorder.js";

// All mutation is local to endpoint nodes; the native graph still owns links and undo.
export function installEndpoint(node) {
  const commit = (fn) => {
    node.graph?.beforeChange();
    fn();
    node.graph?.afterChange();
  };
  const grow = (type, name, widget) => {
    if (
      !MATERIAL_TYPES.has(type) &&
      !(type === POSITION && node.type === INPUTS)
    )
      return false;
    const ps = entries(node);
    const p = {
      id: crypto.randomUUID(),
      slot: ps.length,
      name: name || type,
      type,
      kind: type === POSITION ? "position" : "value",
    };
    if (widget && type === "STRING") {
      p.default = widget.value;
    }
    ps.push(p);
    syncPorts(node, ps);
    return true;
  };
  node.onConnectInput = function (slot, type, output, source) {
    if (this.inputs[slot]?._append) {
      if (!MATERIAL_TYPES.has(type)) return false;
      return grow(type, output.label || output.name);
    }
    return true;
  };
  node.onConnectOutput = function (slot, type, input, target) {
    if (this.outputs[slot]?._append) {
      const widget = target.widgets?.find((w) => w.name === input.name);
      return grow(
        type,
        input.name === "position" ? target.title : input.label || input.name,
        widget,
      );
    }
    return true;
  };
  const rename = (id) => {
    const port = entries(node).find((p) => p.id === id);
    node.graph.list_of_graphcanvas[0].prompt("端口名称", port.name, (value) => {
      if (!value?.trim()) return;
      commit(() => {
        const ps = entries(node);
        ps.find((p) => p.id === id).name = value.trim();
        syncPorts(node, ps);
      });
    });
  };
  node.portReorder = new PortReorder(node, rename);
  node.getExtraMenuOptions = function (_, options) {
    options.push({
      content: "端口设置",
      submenu: {
        options: entries(this).map((p, index) => ({
          content: p.name,
          submenu: {
            options: [
              {
                content: "重命名",
                callback: () => rename(p.id),
              },
              ...[-1, 1]
                .filter(
                  (delta) =>
                    index + delta >= 0 && index + delta < entries(this).length,
                )
                .map((delta) => ({
                  content: delta < 0 ? "上移" : "下移",
                  callback: () =>
                    commit(() => {
                      const ps = entries(this);
                      [ps[index], ps[index + delta]] = [
                        ps[index + delta],
                        ps[index],
                      ];
                      syncPorts(this, ps);
                    }),
                })),
              {
                content: "删除",
                callback: () =>
                  commit(() =>
                    syncPorts(
                      this,
                      entries(this).filter((v) => v.id !== p.id),
                    ),
                  ),
              },
            ],
          },
        })),
      },
    });
  };
}
