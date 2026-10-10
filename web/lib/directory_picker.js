// One server-directory dialog shared by create/open project commands.
export class DirectoryPicker {
  constructor(request, prompt, confirm) {
    this.request = request;
    this.prompt = prompt;
    this.confirm = confirm;
  }
  async choose(create) {
    const dialog = document.createElement("dialog");
    dialog.className =
      "comfy-dialog comfyui-dialog comfy-modal turing-directory-picker";
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    dialog.style.cssText =
      "width:min(900px,92vw);max-height:80vh;padding:16px;background:var(--comfy-menu-bg,#222);color:var(--input-text,#eee);border:1px solid var(--border-color,#555);border-radius:8px";
    const heading = document.createElement("h3");
    heading.textContent = create
      ? "新建画布 · 选择空文件夹"
      : "打开画布 · 选择项目文件夹";
    const path = document.createElement("input");
    path.type = "text";
    path.placeholder = "output 内的相对目录";
    path.style.cssText = "width:100%;box-sizing:border-box;margin:12px 0";
    const columns = document.createElement("div");
    columns.style.cssText =
      "display:grid;grid-template-columns:1fr 1.3fr 1fr;height:min(360px,45vh);gap:8px";
    const panels = ["上一级", "当前目录", "目录信息"].map((title) => {
      const panel = document.createElement("section");
      panel.style.cssText =
        "overflow:auto;border:1px solid var(--border-color,#555);padding:8px;min-width:0";
      const label = document.createElement("div");
      label.textContent = title;
      const body = document.createElement("div");
      panel.append(label, body);
      columns.append(panel);
      return body;
    });
    const status = document.createElement("p");
    status.setAttribute("role", "status");
    const footer = document.createElement("div");
    footer.style.cssText = "display:flex;gap:8px;justify-content:flex-end";
    const addButton = (label, action) => {
      const button = document.createElement("button");
      button.className = "comfy-btn";
      button.textContent = label;
      button.onclick = async () => {
        try {
          await action();
        } catch (e) {
          status.textContent = e.message;
        }
      };
      footer.append(button);
      return button;
    };
    let current = "",
      selected = "",
      info = null,
      serial = 0,
      suspended = false;
    const ask = async (question) => {
      // Comfy's Vue prompt/confirm is outside this native modal. Release the
      // top layer while asking, otherwise the official dialog would be inert.
      suspended = true;
      dialog.style.display = "none";
      const closed = new Promise((resolve) =>
        dialog.addEventListener("close", resolve, { once: true }),
      );
      dialog.close();
      await closed;
      try {
        return await question();
      } finally {
        suspended = false;
        dialog.showModal();
        dialog.style.display = "block";
      }
    };
    const renderList = (panel, items, choose) => {
      panel.replaceChildren();
      for (const item of items) {
        const button = document.createElement("button");
        button.className = "comfy-btn";
        button.style.cssText =
          "display:block;text-align:left;width:100%;margin-top:4px;overflow:hidden;text-overflow:ellipsis";
        button.textContent = (item.directory ? "▸ " : "") + item.name;
        button.onclick = () =>
          choose(item).catch((e) => {
            status.textContent = e.message;
          });
        button.ondblclick = () => {
          if (item.directory)
            navigate(item.path).catch((e) => {
              status.textContent = e.message;
            });
        };
        panel.append(button);
      }
    };
    const preview = async (directory) => {
      const token = ++serial;
      const result = await this.request("projects", { path: directory });
      if (token !== serial || !dialog.open) return;
      selected = directory;
      info = result;
      panels[2].textContent = `${directory || "output"}\n${result.empty ? "空目录" : `${result.items.length} 个可见条目`}\n${result.project ? "Canvas 项目" : "普通目录"}`;
      panels[2].style.whiteSpace = "pre-wrap";
      ok.disabled = !directory || (create ? !result.empty : !result.project);
      remove.disabled =
        !directory || result.project || result.items.some((i) => !i.directory);
      status.textContent =
        create && !result.empty
          ? "新建项目需要空文件夹"
          : !create && !result.project
            ? "请选择包含 canvas.json 的项目文件夹"
            : "";
    };
    const navigate = async (directory) => {
      current = directory;
      path.value = current;
      const result = await this.request("projects", { path: current });
      if (current !== directory || !dialog.open) return;
      renderList(panels[1], result.items, async (item) => {
        if (item.directory) return preview(item.path);
        ++serial;
        selected = "";
        info = null;
        ok.disabled = true;
        remove.disabled = true;
        panels[2].textContent = `${item.name}\n${item.bytes.toLocaleString()} bytes\n${item.path}`;
        panels[2].style.whiteSpace = "pre-wrap";
        status.textContent = "请选择项目文件夹";
      });
      const parent = current.split("/").slice(0, -1).join("/");
      if (current) {
        const listing = await this.request("projects", { path: parent });
        if (current !== directory || !dialog.open) return;
        renderList(
          panels[0],
          [
            { name: "..", path: parent, directory: true },
            ...listing.items.filter((i) => i.directory),
          ],
          (item) => navigate(item.path),
        );
      } else panels[0].replaceChildren();
      await preview(current);
    };
    addButton("上一级", () => navigate(current.split("/").slice(0, -1).join("/")));
    addButton("新建文件夹", async () => {
      const name = await ask(() => this.prompt("新文件夹名称"));
      if (!name) return;
      if (/[\\/]/.test(name) || [".", ".."].includes(name))
        throw Error("请输入单个文件夹名称");
      const directory = [current, name].filter(Boolean).join("/");
      await this.request("directory/create", { directory });
      await navigate(directory);
    });
    const remove = addButton("删除空目录", async () => {
      if (!selected || info?.project) return;
      if (
        !(await ask(() =>
          this.confirm(
            `删除空目录 output/${selected}？仅无文件的目录树可删除，删除后不可恢复。`,
          ),
        ))
      )
        return;
      await this.request("directory/delete-empty", { directory: selected });
      await navigate(selected.split("/").slice(0, -1).join("/"));
    });
    addButton("取消", () => dialog.close());
    const ok = addButton(create ? "创建项目" : "打开项目", () =>
      dialog.close(selected),
    );
    ok.disabled = true;
    dialog.append(heading, path, columns, status, footer);
    document.body.append(dialog);
    const result = new Promise((resolve) => {
      dialog.addEventListener("close", () => {
        if (suspended) return;
        ++serial;
        resolve(dialog.returnValue || null);
        dialog.remove();
      });
    });
    path.onkeydown = (event) => {
      if (event.key === "Enter")
        navigate(path.value.trim()).catch((e) => {
          status.textContent = e.message;
        });
    };
    dialog.showModal();
    // comfy-modal is display:none until Comfy's own dialog service toggles it;
    // this native HTML dialog is opened with showModal instead.
    dialog.style.display = "block";
    await navigate("").catch((e) => {
      status.textContent = e.message;
    });
    return result;
  }
}
