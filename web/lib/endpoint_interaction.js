import { POSITION, entries, syncPorts } from "./canvas_ports.js";

// All mutation is local to endpoint nodes; the native graph still owns links and undo.
export function installEndpoint(node) {
  const commit = (fn) => {
    node.graph?.beforeChange();
    fn();
    node.graph?.afterChange();
  };
  const grow = (type, name, widget) => {
    const ps = entries(node);
    const p = {
      id: crypto.randomUUID(),
      slot: ps.length,
      name: name || type,
      type: Array.isArray(type) ? "COMBO" : type,
      kind: type === POSITION ? "position" : "value",
    };
    if (widget) {
      p.default = widget.value;
      if (p.type === "COMBO") {
        const values = widget.options?.values;
        p.options = typeof values === "function" ? values() : values;
      }
    }
    if (Array.isArray(type)) p.options = type;
    ps.push(p);
    syncPorts(node, ps);
  };
  node.onConnectInput = function (slot, type, output, source) {
    if (this.inputs[slot]?._append) {
      grow(type, output.label || output.name);
    }
    return true;
  };
  node.onConnectOutput = function (slot, type, input, target) {
    if (this.outputs[slot]?._append) {
      const widget = target.widgets?.find((w) => w.name === input.name);
      grow(
        type,
        input.name === "position" ? target.title : input.label || input.name,
        widget,
      );
    }
    return true;
  };
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
                callback: () =>
                  this.graph.list_of_graphcanvas[0].prompt(
                    "端口名称",
                    p.name,
                    (value) =>
                      commit(() => {
                        const ps = entries(this);
                        ps[index].name = value;
                        syncPorts(this, ps);
                      }),
                  ),
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
