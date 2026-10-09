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
  page.on("pageerror", (e) => console.log("PAGE ERROR", e.message));
  page.on("console", (e) => {
    if (e.type() === "error") console.log("CONSOLE", e.text());
  });
  await page.goto("http://127.0.0.1:18188");
  await page.waitForFunction(
    () =>
      window.app?.graph &&
      window.LiteGraph?.registered_node_types.TuringCanvasProject,
  );
  const result = await page.evaluate(async () => {
    const { app } = await import("/scripts/app.js");
    const m =
      await import("/extensions/comfyui-turing-utils/material_workspace.js");
    const directory = `native-smoke-${Date.now()}`;
    await m.request("directory/create", { directory });
    await m.openProject(directory, true);
    if (app.graph._nodes.length !== 1) throw Error("Missing root");
    await m.request("card/add", { directory, kind: "text" });
    await m.openProject(directory);
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
    await m.request("text", {
      directory,
      node: material.id,
      text: "persisted",
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
      target.inputs.findIndex((i) => i.name === "value"),
    );
    const task = await app.graphToPrompt(),
      name = `native-${Date.now()}.json`;
    await m.request("template/save", {
      name,
      workflow: task.workflow,
      prompt: task.output,
    });
    await m.request("card/add", { directory, name });
    await m.openProject(directory);
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
      const state = await m.request("project", { directory });
      if (state.selections[output.id]?.text === "computed") {
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
    const linkedTarget = app.graph._nodes.find(
      (n) => n.type === "TuringMaterialText",
    );
    const port = addPort(endpoint, "source", "STRING");
    endpoint.connect(
      port.slot,
      linkedTarget,
      linkedTarget.inputs.findIndex((i) => i.name === "value"),
    );
    const linked = await app.graphToPrompt(),
      linkedName = `linked-${Date.now()}.json`;
    await m.request("template/save", {
      name: linkedName,
      workflow: linked.workflow,
      prompt: linked.output,
    });
    const added = await m.request("card/add", { directory, name: linkedName });
    await m.openProject(directory);
    const from = app.graph._nodes.find(
      (n) => n.properties.instance === card.properties.instance,
    );
    const to = app.graph._nodes.find((n) => n.properties.instance === added.id);
    from.connect(0, to, 0);
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
      const state = await m.request("project", { directory });
      if (state.selections[finalId]?.text === "persisted") {
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
    };
  });
  console.log(JSON.stringify(result));
  assert.equal(result.dialog, "function");
  if (process.env.MATERIAL_VIDEO_FIXTURE) {
    const bytes = await readFile(process.env.MATERIAL_VIDEO_FIXTURE);
    const media = await page.evaluate(
      async ({ directory, base64 }) => {
        const { app } = await import("/scripts/app.js"),
          m =
            await import("/extensions/comfyui-turing-utils/material_workspace.js");
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
        return { coldVideo: true, releaseOnTabSwitch: true };
      },
      { directory: result.directory, base64: bytes.toString("base64") },
    );
    console.log(JSON.stringify(media));
  }
} finally {
  await browser.close();
}
