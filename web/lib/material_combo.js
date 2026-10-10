const lists = new Map();

export function bindMaterialCombo(node, widget, { key, list, selected, select }) {
  let entry = lists.get(key);
  if (!entry)
    lists.set(
      key,
      (entry = { items: [], listeners: new Set(), load: null, loaded: false, list }),
    );
  const update = () => {
    widget.options.values = [...new Set(["", ...entry.items.map((i) => i.name)])];
    widget.value = selected()?.split("/").at(-1) || "";
    if (widget.value && !widget.options.values.includes(widget.value))
      widget.options.values.push(widget.value);
    node.graph?.setDirtyCanvas(true, true);
  };
  entry.listeners.add(update);
  const remove = widget.onRemove;
  widget.onRemove = () => {
    entry.listeners.delete(update);
    if (!entry.listeners.size) lists.delete(key);
    remove?.call(widget);
  };
  widget.callback = async (name) => {
    const item = entry.items.find((i) => i.name === name);
    if (!item) {
      update();
      return;
    }
    try {
      await select(item.id);
    } finally {
      update();
    }
  };
  update();
  widget.refreshMaterialList = () => refreshMaterialList(key);
  return entry.loaded ? Promise.resolve() : refreshMaterialList(key);
}

export async function refreshMaterialList(key) {
  const entry = lists.get(key);
  if (!entry) return;
  if (!entry.load)
    entry.load = entry
      .list()
      .then((items) => {
        entry.items = items;
        entry.loaded = true;
        for (const update of entry.listeners) update();
      })
      .finally(() => {
        entry.load = null;
      });
  return entry.load;
}
