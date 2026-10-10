import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { readFile } from "node:fs/promises";
const { chromium } = await import(
  pathToFileURL(process.env.PLAYWRIGHT_MODULE).href
);
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("response", async (response) => {
    if (
      response.status() >= 400 &&
      response.url().includes("/turing/workspace/")
    )
      console.log("API ERROR", response.url(), await response.text());
  });
  page.on("pageerror", (e) => {
    errors.push(e.message);
    console.log("PAGE ERROR", e.message);
  });
  page.on("console", (e) => {
    if (e.type() === "error") {
      console.log("CONSOLE", e.text());
      if (e.text().includes("Error calling extension 'TuringUtils"))
        errors.push(e.text());
    }
  });
  await page.goto("http://127.0.0.1:18188");
  await page.waitForFunction(
    () =>
      window.app?.graph &&
      window.LiteGraph?.registered_node_types.TuringCanvasProject,
  );
  await page.waitForFunction(
    () =>
      window.app.extensionManager.workflow.activeWorkflow?.isLoaded &&
      !window.app.extensionManager.workflow.isBusy,
  );
  const result = await page.evaluate(async () => {
    const { app } = await import("/scripts/app.js");
    const m =
      await import("/extensions/comfyui-turing-utils/material_workspace.js");
    const directory = `native-smoke-${Date.now()}`;
    await m.request("directory/create", { directory });
    await m.openProject(directory, true);
    if (app.graph._nodes.length !== 1) throw Error("Missing root");
    await m.addCard({ kind: "text" });
    const card = app.graph._nodes.find((n) => n.type === "TuringCanvasCard");
    if (!card.widgets?.length)
      throw Error(
        "Missing native widgets: " +
          JSON.stringify({
            properties: card.properties,
            ctor: card.constructor.name,
            configure: typeof card.onConfigure,
          }),
      );
    const w = card.widgets.find(
      (w) => w.type === "customtext" || w.inputEl?.tagName === "TEXTAREA",
    );
    const material = card.properties.materials[0];
    card.trims = { [material.id]: { start: 0.2, end: 0.8 } };
    const widgets = card.widgets;
    const trims = card.trims;
    await m.addCard({ kind: "text" }, [900, 230]);
    if (app.graph.getNodeById(card.id) !== card || card.widgets !== widgets ||
        card.trims !== trims || !card.widgets.includes(w))
      throw Error("Adding a card rebuilt existing nodes or session state");
    await m.request("select", {
      directory,
      node: material.id,
      selection: { text: "persisted" },
      revision: material.selection.revision,
    });
    card.pos = [530, 230];
    await m.saveProject();
    const raw = await m.request("project/open", { directory });
    if (raw.workflow.nodes[1].pos[0] !== 530) throw Error("Layout not saved");
    const starter = await (
      await fetch("/turing/workspace/new-template")
    ).json();
    await m.openTab(starter, `new-card-${Date.now()}.json`);
    if (app.graph._nodes.length !== 6) throw Error("Starter missing");
    for (const type of [
      "TuringMaterialImage",
      "TuringMaterialVideo",
      "TuringMaterialAudio",
    ]) {
      const node = app.graph._nodes.find((n) => n.type === type);
      if (
        !node.widgets.some(
          (w) => w.type === "combo" && ["file", "audio"].includes(w.name),
        )
      )
        throw Error("Material has no native file combo");
      if (
        !node.widgets.some(
          (w) => w.type === "button" && w.label === "choose file to upload",
        )
      )
        throw Error("Material has no native upload button");
    }
    const prompt = await app.graphToPrompt();
    if (Object.keys(prompt.output).length !== 6)
      throw Error("Ordinary workflow broken");
    const beforeTabs = app.extensionManager.workflow.openWorkflows.length;
    await m.openProject(directory);
    if (app.extensionManager.workflow.openWorkflows.length !== beforeTabs)
      throw Error("Project duplicated its tab");
    await m.openTab(starter, `compute-${Date.now()}.json`);
    const target = app.graph._nodes.find(
      (n) => n.type === "TuringMaterialText",
    );
    const constant = LiteGraph.createNode("PrimitiveString");
    app.graph.add(constant);
    constant.widgets.find((w) => w.name === "value").value = "computed";
    constant.connect(
      0,
      target,
      target.inputs.findIndex((i) => i.name === "text"),
    );
    if (target.widgets.find((w) => w.name === "text").type !== "customtext")
      throw Error("Connected text widget disappeared");
    const task = await app.graphToPrompt(),
      name = `native-${Date.now()}.json`;
    await m.request("template/save", {
      name,
      workflow: task.workflow,
      prompt: task.output,
    });
    await m.openProject(directory);
    await m.addCard({ name });
    const generated = app.graph._nodes.find(
      (n) =>
        n.type === "TuringCanvasCard" &&
        n.properties.materials.some((x) => x.executable),
    );
    const output = generated.properties.materials.find(
      (x) => x.kind === "text",
    );
    await generated.widgets
      .find((w) => w.name.includes("执行到这里"))
      .callback();
    let success = false;
    for (let i = 0; i < 60; i++) {
      const state = await m.request("selection", { directory, node: output.id });
      if (state.text === "computed") {
        success = true;
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    if (!success) throw Error("Local execution did not publish text");
    await m.openTab(task.workflow, `linked-${Date.now()}.json`);
    const { addPort } =
      await import("/extensions/comfyui-turing-utils/lib/canvas_ports.js");
    const endpoint = app.graph._nodes.find(
      (n) => n.type === "TuringCanvasInputs",
    );
    if (!endpoint.inputs.some((s) => s._append))
      throw Error("Missing dynamic source input");
    const beforePorts = endpoint.outputs.length;
    if (
      endpoint.onConnectInput(
        endpoint.inputs.findIndex((s) => s._append),
        "INT",
        { name: "integer" },
        constant,
      ) !== false ||
      endpoint.outputs.length !== beforePorts
    )
      throw Error("Endpoint accepted a non-material type");
    const linkedTarget = app.graph._nodes.find(
      (n) => n.type === "TuringMaterialText",
    );
    const port = addPort(endpoint, "source", "STRING");
    endpoint.connect(
      port.slot,
      linkedTarget,
      linkedTarget.inputs.findIndex((i) => i.name === "text"),
    );
    const linked = await app.graphToPrompt(),
      linkedName = `linked-${Date.now()}.json`;
    await m.request("template/save", {
      name: linkedName,
      workflow: linked.workflow,
      prompt: linked.output,
    });
    await m.openProject(directory);
    const submenu = app.canvas.getCanvasMenuOptions()[0].submenu.options;
    const option = submenu.find((o) => o?.content === linkedName);
    if (!option || !submenu.some((o) => o?.content === "图片"))
      throw Error("Missing native Add Node submenu");
    await option.callback();
    const added = {
      id: app.graph._nodes.find(
        (n) => n.properties.card?.title === linkedName.slice(0, -5),
      )?.properties.instance,
    };
    if (!added.id) throw Error("Menu did not add a card instance");
    const from = app.graph._nodes.find(
      (n) => n.properties.instance === card.properties.instance,
    );
    const to = app.graph._nodes.find((n) => n.properties.instance === added.id);
    from.connect(0, to, 0);
    const fromWidgets = from.widgets, toWidgets = to.widgets,
      link = to.inputs[0].link;
    const imageNode = app.graph._nodes.find((n) =>
      n.properties.materials?.some((m) => m.kind === "image"));
    const image = imageNode.properties.materials.find((m) => m.kind === "image");
    await imageNode.widgets.find((w) => w.name === image.id).refreshMaterialList();
    const { api } = await import("/scripts/api.js");
    const fetchApi = api.fetchApi;
    let historyReads = 0;
    api.fetchApi = function (path, ...args) {
      if (path === "/turing/workspace/history") historyReads++;
      return fetchApi.call(this, path, ...args);
    };
    try {
      await m.addCard({ kind: "image" }, [1300, 230]);
      if (historyReads) throw Error("Adding a card reloaded an existing history list");
    } finally {
      api.fetchApi = fetchApi;
    }
    if (app.graph.getNodeById(from.id) !== from || app.graph.getNodeById(to.id) !== to ||
        from.widgets !== fromWidgets || to.widgets !== toWidgets || to.inputs[0].link !== link)
      throw Error("Incremental insertion changed existing widgets or links");
    await m.saveProject();
    await m.openProject(directory);
    const restored = app.graph._nodes.find(
      (n) => n.properties.instance === added.id,
    );
    if (restored.inputs[0].link === null)
      throw Error("Cross-card link not restored");
    await restored.widgets
      .find((w) => w.name.includes("执行到这里"))
      .callback();
    const finalId = restored.properties.materials.find(
      (x) => x.kind === "text",
    ).id;
    let linkedSuccess = false;
    for (let i = 0; i < 60; i++) {
      const state = await m.request("selection", { directory, node: finalId });
      if (state.text === "persisted") {
        linkedSuccess = true;
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    if (!linkedSuccess) throw Error("Cross-card execution ignored source");
    return {
      directory,
      nativeWidgets: true,
      tabs: app.extensionManager.workflow.openWorkflows.length,
      dialog: typeof app.extensionManager.dialog.prompt,
      localExecution: success,
      crossCardExecution: linkedSuccess,
      incrementalInsertion: true,
      sharedHistoryList: true,
    };
  });
  console.log(JSON.stringify(result));
  const interactions = await page.evaluate(async () => {
    const { app } = await import("/scripts/app.js");
    const m =
      await import("/extensions/comfyui-turing-utils/material_workspace.js");
    const { addPort, entries } =
      await import("/extensions/comfyui-turing-utils/lib/canvas_ports.js");
    const starter = await (
      await fetch("/turing/workspace/new-template")
    ).json();
    await m.openTab(starter, `port-test-${Date.now()}.json`);
    const checkLabels = (node) => {
      for (const port of entries(node)) {
        const output = node.outputs[port.slot];
        if (output.renderingLabel !== port.name)
          throw Error(`${node.type} output label differs from ${port.name}`);
        const input = node.inputs.find((s) => s._portId === port.id);
        if (input && input.renderingLabel !== port.name)
          throw Error(`${node.type} input label differs from ${port.name}`);
      }
    };
    const endpoints = app.graph._nodes.filter((n) =>
      ["TuringCanvasInputs", "TuringCanvasOutputs"].includes(n.type),
    );
    endpoints.forEach(checkLabels);
    const endpoint = app.graph._nodes.find(
      (n) => n.type === "TuringCanvasInputs",
    );
    const a = addPort(endpoint, "A", "STRING"),
      b = addPort(endpoint, "B", "STRING");
    const first = LiteGraph.createNode("PrimitiveString"),
      second = LiteGraph.createNode("PrimitiveString");
    app.graph.add(first);
    app.graph.add(second);
    first.widgets[0].value = "first";
    second.widgets[0].value = "second";
    first.connect(
      0,
      endpoint,
      endpoint.inputs.findIndex((s) => s._portId === a.id),
    );
    second.connect(
      0,
      endpoint,
      endpoint.inputs.findIndex((s) => s._portId === b.id),
    );
    const text = app.graph._nodes.find((n) => n.type === "TuringMaterialText");
    endpoint.connect(
      a.slot,
      text,
      text.inputs.findIndex((s) => s.name === "text"),
    );
    const y = endpoint.getConnectionPos(false, a.slot)[1] - endpoint.pos[1];
    endpoint.onMouseDown(
      { button: 0, detail: 1 },
      [endpoint.size[0] / 2, y],
      app.canvas,
    );
    if (!endpoint.portReorder.drag) throw Error("Port label cannot be dragged");
    endpoint.onMouseMove(
      {},
      [endpoint.size[0] / 2, y + LiteGraph.NODE_SLOT_HEIGHT],
      app.canvas,
    );
    await new Promise((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(resolve)),
    );
    if (!endpoint.outputs.some((s) => s.pos)) throw Error("No drag animation");
    endpoint.onMouseUp(
      {},
      [endpoint.size[0] / 2, y + LiteGraph.NODE_SLOT_HEIGHT],
      app.canvas,
    );
    if (entries(endpoint).at(-1).id !== a.id)
      throw Error("Port order not changed");
    for (const input of endpoint.inputs.filter((s) => !s._append)) {
      const output = endpoint.outputs.findIndex(
        (s) => s._portId === input._portId,
      );
      if (
        Math.abs(
          endpoint.getConnectionPos(true, endpoint.inputs.indexOf(input))[1] -
            endpoint.getConnectionPos(false, output)[1],
        ) > 0.1
      )
        throw Error("Paired ports misaligned");
    }
    const prompt = await app.graphToPrompt();
    const p = prompt.output[String(endpoint.id)].inputs,
      ordered = entries(endpoint);
    if (
      p[`port_${ordered.find((x) => x.id === a.id).slot}`][0] !==
      String(first.id)
    )
      throw Error("Reorder changed input source");
    if (
      prompt.output[String(text.id)].inputs.text[1] !==
      ordered.find((x) => x.id === a.id).slot
    )
      throw Error("Reorder changed downstream source");
    endpoint.portReorder.begin(ordered.length - 1);
    endpoint.portReorder.drag.target = 0;
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    if (entries(endpoint).at(-1).id !== a.id || endpoint.portReorder.drag)
      throw Error("Escape did not cancel drag");
    const oldPrompt = app.canvas.prompt;
    let renamed = false;
    app.canvas.prompt = (title, value, change) => {
      renamed = true;
      change("Renamed");
    };
    try {
      const row = entries(endpoint).findIndex((p) => p.id === a.id);
      const y = endpoint.getConnectionPos(false, row)[1] - endpoint.pos[1];
      endpoint.onDblClick({}, [endpoint.size[0] / 2, y], app.canvas);
    } finally {
      app.canvas.prompt = oldPrompt;
    }
    if (
      !renamed ||
      entries(endpoint).find((p) => p.id === a.id).name !== "Renamed"
    )
      throw Error("Native double-click did not rename the port");
    endpoints.forEach(checkLabels);
    const saved = app.graph.serialize();
    for (const node of saved.nodes.filter((n) =>
      ["TuringCanvasInputs", "TuringCanvasOutputs"].includes(n.type),
    )) {
      // Serialized slots can retain translated names from the old schema.
      for (const slot of [...node.inputs, ...node.outputs]) {
        slot.localized_name = "*";
        slot.label = "stale label";
      }
    }
    await app.loadGraphData(saved);
    const restored = app.graph._nodes.filter((n) =>
      ["TuringCanvasInputs", "TuringCanvasOutputs"].includes(n.type),
    );
    restored.forEach(checkLabels);
    const linked = await app.graphToPrompt();
    if (
      linked.output[String(endpoint.id)].inputs[
        `port_${ordered.find((p) => p.id === a.id).slot}`
      ][0] !== String(first.id) ||
      linked.output[String(text.id)].inputs.text[1] !==
        ordered.find((p) => p.id === a.id).slot
    )
      throw Error("Label synchronization changed links on reload");
    return {
      dragReorder: true,
      pairedPorts: true,
      materialTypeGuard: true,
      cancelDrag: true,
      doubleClickRename: true,
      endpointLabels: true,
      restoredLabels: true,
    };
  });
  console.log(JSON.stringify(interactions));
  const pickerDirectory = `picker-smoke-${Date.now()}`;
  await page.evaluate(async (directory) => {
    const { request } =
      await import("/extensions/comfyui-turing-utils/material_workspace.js");
    const { DirectoryPicker } =
      await import("/extensions/comfyui-turing-utils/lib/directory_picker.js");
    await request("directory/create", { directory });
    window.pickerResult = new DirectoryPicker(
      request,
      async () => null,
      async () => false,
    ).choose(true);
  }, pickerDirectory);
  await page.locator(".turing-directory-picker").waitFor({ state: "visible" });
  await page
    .getByRole("button", { name: `▸ ${pickerDirectory}`, exact: true })
    .dispatchEvent("click");
  await page.waitForFunction(() =>
    [...document.querySelectorAll(".turing-directory-picker button")].some(
      (b) => b.textContent === "创建项目" && !b.disabled,
    ),
  );
  await page
    .getByRole("button", { name: "创建项目", exact: true })
    .dispatchEvent("click");
  assert.equal(await page.evaluate(() => window.pickerResult), pickerDirectory);
  console.log(
    JSON.stringify({ directoryColumns: true, emptyFolderSelection: true }),
  );
  await page.evaluate(async (directory) => {
    const { request } =
      await import("/extensions/comfyui-turing-utils/material_workspace.js");
    const { DirectoryPicker } =
      await import("/extensions/comfyui-turing-utils/lib/directory_picker.js");
    window.folderPromptReleasedModal = false;
    window.pickerResult = new DirectoryPicker(
      request,
      async () => {
        window.folderPromptReleasedModal = !document.querySelector(
          ".turing-directory-picker",
        ).open;
        return directory;
      },
      async () => false,
    ).choose(true);
  }, `nested-picker-${Date.now()}`);
  await page
    .getByRole("button", { name: "新建文件夹", exact: true })
    .dispatchEvent("click");
  await page.waitForFunction(
    () =>
      window.folderPromptReleasedModal &&
      [...document.querySelectorAll(".turing-directory-picker button")].some(
        (b) => b.textContent === "创建项目" && !b.disabled,
      ),
  );
  await page
    .getByRole("button", { name: "取消", exact: true })
    .dispatchEvent("click");
  assert.equal(await page.evaluate(() => window.pickerResult), null);
  const persistence = await page.evaluate(async (directory) => {
    const { app } = await import("/scripts/app.js");
    const m =
      await import("/extensions/comfyui-turing-utils/material_workspace.js");
    await m.openProject(directory);
    const root = app.graph._nodes.find((n) => n.type === "TuringCanvasProject");
    if (root.widgets.find((w) => w.name === "素材像素上限（MP）").value !== 4)
      throw Error("Incorrect default pixel limit");
    await Promise.all([
      root.widgets.find((w) => w.name === "项目名称").callback("Browser smoke"),
      root.widgets.find((w) => w.name === "素材像素上限（MP）").callback(3),
    ]);
    const media = app.graph._nodes.find((n) =>
      n.properties.materials?.some((m) => m.kind === "text"),
    );
    media.pos = [710, 270];
    const text = media.properties.materials.find((m) => m.kind === "text");
    await media.widgets.find((w) => w.name === text.id).callback("autosaved");
    let data = await m.request("project/open", { directory });
    if (
      data.workflow.nodes.find(
        (n) => n.properties.instance === media.properties.instance,
      ).pos[0] !== 710
    )
      throw Error("Text publication did not autosave layout");
    if (data.workflow.nodes[0].properties.settings.name !== "Browser smoke")
      throw Error("Global settings did not persist");
    if (data.workflow.nodes[0].properties.settings.max_megapixels !== 3)
      throw Error("Concurrent settings lost a change");
    if (
      !root.widgets.find((w) => w.name === "画布统计").value.includes("text:")
    )
      throw Error("Missing material statistics");
    await Promise.all([m.saveProject(), m.saveProject(), m.saveProject()]);
    const nodeCount = app.graph._nodes.length;
    const starter = await (
      await fetch("/turing/workspace/new-template")
    ).json();
    await m.openTab(starter, `ordinary-${Date.now()}.json`);
    const ordinary = app.graph;
    await m.saveProject(directory);
    if (app.graph !== ordinary || app.graph.extra.turing_project)
      throw Error("Background save switched the active tab");
    data = await m.request("project/open", { directory });
    if (
      data.workflow.nodes.find(
        (n) => n.properties.instance === media.properties.instance,
      ).pos[0] !== 710 ||
      data.workflow.nodes.length !== nodeCount
    )
      throw Error("Background save used the wrong workflow");
    await m.openProject(directory);
    const { api } = await import("/scripts/api.js");
    const fetchApi = api.fetchApi;
    let release, arrived;
    const gate = new Promise((r) => { release = r; });
    const ready = new Promise((r) => { arrived = r; });
    api.fetchApi = async function (path, ...args) {
      const response = await fetchApi.call(this, path, ...args);
      if (path === "/turing/workspace/card/add") {
        arrived();
        await gate;
      }
      return response;
    };
    let added;
    try {
      const pending = m.addCard({ kind: "text" }, [1400, 350]);
      await ready;
      await m.openTab(starter, `switch-during-add-${Date.now()}.json`);
      const active = app.graph.serialize();
      release();
      added = await pending;
      if (JSON.stringify(app.graph.serialize()) !== JSON.stringify(active))
        throw Error("Background insertion changed the active workflow");
    } finally {
      release();
      api.fetchApi = fetchApi;
    }
    data = await m.request("project/open", { directory });
    if (data.workflow.nodes.length !== nodeCount + 1 ||
        data.workflow.nodes.find((n) => n.properties.instance === added.id)?.pos[0] !== 1400)
      throw Error("Background insertion lost the new card or position");
    await m.openProject(directory);
    if (!app.graph._nodes.find((n) => n.properties.instance === added.id)?.widgets.length)
      throw Error("Background card did not restore native widgets");
    return {
      autosave: true,
      inactiveProjectSave: true,
      inactiveProjectInsertion: true,
      concurrentSaves: true,
      globalSettings: true,
      statistics: true,
    };
  }, result.directory);
  console.log(JSON.stringify(persistence));
  assert.equal(result.dialog, "function");
  if (process.env.MATERIAL_VIDEO_FIXTURE) {
    const bytes = await readFile(process.env.MATERIAL_VIDEO_FIXTURE);
    const media = await page.evaluate(
      async ({ directory, base64 }) => {
        const { app } = await import("/scripts/app.js"),
          m =
            await import("/extensions/comfyui-turing-utils/material_workspace.js");
        await m.openProject(directory);
        const form = new FormData();
        form.append(
          "file",
          new Blob([Uint8Array.from(atob(base64), (c) => c.charCodeAt(0))]),
          "sample.webm",
        );
        const response = await fetch(
          `/turing/workspace/import?directory=${directory}&kind=video`,
          { method: "POST", body: form },
        );
        if (!response.ok) throw Error(await response.text());
        const { asset } = await response.json();
        const card = app.graph._nodes.find(
          (n) =>
            n.type === "TuringCanvasCard" &&
            n.properties.materials.some((m) => m.kind === "video"),
        );
        const material = card.properties.materials.find(
          (m) => m.kind === "video",
        );
        await m.request("select", {
          directory,
          node: material.id,
          selection: { asset },
          revision: 0,
        });
        await m.openProject(directory);
        const node = app.graph._nodes.find(
          (n) => n.properties?.instance === card.properties.instance,
        );
        const file = node.widgets.find((w) => w.name === material.id);
        await file.refreshMaterialList();
        if (!file.options.values.includes(asset.split("/").at(-1)))
          throw Error("History dropdown did not use filenames");
        await file.callback(asset.split("/").at(-1));
        const ranged = await fetch(
          `/turing/workspace/asset?${new URLSearchParams({ directory, asset })}`,
          { headers: { Range: "bytes=0-127" } },
        );
        if (
          ranged.status !== 206 ||
          (await ranged.arrayBuffer()).byteLength !== 128
        )
          throw Error("Media streaming does not support byte ranges");
        if (node.player) throw Error("Player created before request");
        await node.widgets.find((w) => w.name === "video · 预览").callback();
        if (!node.player || node.player.preload !== "none")
          throw Error("Player not lazy");
        const player = node.player;
        const source = await (
          await fetch("/turing/workspace/new-template")
        ).json();
        await m.openTab(source, `media-test-${Date.now()}.json`);
        if (player.getAttribute("src"))
          throw Error("Switching tab retained media source");
        return {
          coldVideo: true,
          releaseOnTabSwitch: true,
          filenameHistory: true,
          rangeStreaming: true,
        };
      },
      { directory: result.directory, base64: bytes.toString("base64") },
    );
    console.log(JSON.stringify(media));
  }
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
}
