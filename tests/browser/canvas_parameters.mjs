import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

const { chromium } = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href);
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(process.env.CANVAS_URL || "http://127.0.0.1:18188");
  await page.waitForFunction(
    () =>
      window.app?.graph && window.LiteGraph?.registered_node_types.TuringCanvasInputs,
  );
  await page.waitForFunction(
    () =>
      window.app.extensionManager.workflow.activeWorkflow?.isLoaded &&
      !window.app.extensionManager.workflow.isBusy,
  );
  const result = await page.evaluate(async () => {
    const { app } = await import("/scripts/app.js");
    const workspace = await import(
      "/extensions/comfyui-turing-utils/material_workspace.js"
    );
    const { entries, syncPorts } = await import(
      "/extensions/comfyui-turing-utils/lib/canvas_ports.js"
    );
    const check = (condition, message) => {
      if (!condition) throw Error(message);
    };
    const stamp = Date.now();
    const directory = `canvas-parameters-${stamp}`;
    const name = `canvas-parameters-${stamp}.json`;
    await workspace.request("directory/create", { directory });
    await workspace.openProject(directory, true);
    const starter = await (await fetch("/turing/workspace/new-template")).json();
    await workspace.openTab(starter, name);
    const endpoint = app.graph._nodes.find((n) => n.type === "TuringCanvasInputs");
    const outputs = app.graph._nodes.find((n) => n.type === "TuringCanvasOutputs");
    const text = app.graph._nodes.find((n) => n.type === "TuringMaterialText");
    const sampler = LiteGraph.createNode("KSampler");
    const boolean = LiteGraph.createNode("PrimitiveBoolean");
    const string = LiteGraph.createNode("PrimitiveString");
    for (const node of [sampler, boolean, string]) app.graph.add(node);
    const targets = [
      [sampler, "steps", "INT"],
      [sampler, "cfg", "FLOAT"],
      [boolean, "value", "BOOLEAN"],
      [string, "value", "STRING"],
      [sampler, "sampler_name", "COMBO"],
    ];
    const ids = {};
    for (const [target, input, type] of targets) {
      const link = endpoint.connect(
        endpoint.outputs.findIndex((s) => s._append),
        target,
        target.inputs.findIndex((s) => s.name === input),
      );
      check(link, `Cannot expose ${type}`);
      const port = entries(endpoint).at(-1);
      check(port.kind === "parameter" && port.type === type, `Wrong ${type} port`);
      ids[type] = port.id;
      const widget = endpoint.widgets.find((w) => w._portId === port.id);
      check(widget && widget.options.serialize === false, `Missing ${type} form`);
      if (type === "FLOAT")
        check(widget.options.step2 === 0.1, "Numeric step not inherited");
      if (type === "COMBO")
        check(widget.options.values.includes("heun"), "Combo choices missing");
    }
    string.connect(
      0,
      text,
      text.inputs.findIndex((s) => s.name === "text"),
    );
    const count = entries(endpoint).length;
    check(
      !endpoint.connect(
        endpoint.outputs.findIndex((s) => s._append),
        sampler,
        sampler.inputs.findIndex((s) => s.name === "model"),
      ),
      "MODEL was accepted",
    );
    check(entries(endpoint).length === count, "Rejected connection changed ports");
    check(
      outputs.onConnectInput(
        outputs.inputs.findIndex((s) => s._append),
        "INT",
        {},
        sampler,
      ) === false,
      "Outputs accepted parameters",
    );
    const ordered = entries(endpoint).reverse();
    syncPorts(endpoint, ordered);
    const edited = endpoint.widgets.find((w) => w._portId === ids.INT);
    edited.value = 11;
    edited.callback(11);
    const snapshot = await app.graphToPrompt();
    const intPort = entries(endpoint).find((p) => p.id === ids.INT);
    check(
      snapshot.output[String(sampler.id)].inputs.steps[1] === intPort.slot,
      "Reorder changed parameter wiring",
    );
    check(
      Object.keys(snapshot.output[String(endpoint.id)].inputs).length === 1,
      "Display widgets leaked into execution inputs",
    );
    await workspace.openTab(snapshot.workflow, `reloaded-${stamp}.json`);
    const restored = app.graph.getNodeById(endpoint.id);
    check(
      restored.widgets.find((w) => w._portId === ids.INT).value === 11,
      "Endpoint default did not survive reload",
    );
    const task = await app.graphToPrompt();
    await workspace.request("template/save", {
      name,
      workflow: task.workflow,
      prompt: task.output,
    });
    const template = await workspace.request("template/open", { name });
    await workspace.openProject(directory);
    const added = await workspace.addCard({ name });
    let card = app.graph._nodes.find((n) => n.properties.instance === added.id);
    check(card.inputs.length === 0, "Parameters became material sockets");
    check(card.properties.card.fields.length === 5, "Missing card form fields");
    const values = {
      INT: 13,
      FLOAT: 1.25,
      BOOLEAN: false,
      STRING: "saved text",
      COMBO: "heun",
    };
    for (const [type, value] of Object.entries(values)) {
      const field = card.properties.card.fields.find((f) => f.input === ids[type]);
      const widget = card.widgets.find((w) => w.name === field.input);
      check(widget, `Missing projected ${type} widget`);
      widget.value = value;
      await widget.callback(value);
    }
    const opened = await workspace.request("card/open", { directory, id: added.id });
    for (const [type, value] of Object.entries(values))
      check(opened.parameters[ids[type]] === value, `${type} value was not persisted`);
    check(
      JSON.stringify((await workspace.request("template/open", { name })).workflow) ===
        JSON.stringify(template.workflow),
      "Instance edits changed template",
    );
    check(
      JSON.stringify(opened.workflow.extra.turing_card.overrides) === "{}",
      "Instance values were stored in workflow rather than canvas",
    );
    await workspace.openProject(directory);
    card = app.graph._nodes.find((n) => n.properties.instance === added.id);
    for (const [type, value] of Object.entries(values))
      check(
        card.properties.values[ids[type]] === value,
        `${type} not restored on project open`,
      );
    await card.widgets.find((w) => w.name === "编辑工作流").callback();
    const editing = app.graph._nodes.find((n) => n.type === "TuringCanvasInputs");
    for (const [type, value] of Object.entries(values))
      check(
        editing.widgets.find((w) => w._portId === ids[type]).value === value,
        `${type} not restored in workflow editor`,
      );
    const intWidget = editing.widgets.find((w) => w._portId === ids.INT);
    intWidget.value = 20;
    intWidget.callback(20);
    const editedTask = await app.graphToPrompt();
    await workspace.request("card/save", {
      directory,
      id: added.id,
      revision: opened.revision,
      workflow: editedTask.workflow,
      prompt: editedTask.output,
      parameters: Object.fromEntries(
        entries(editing)
          .filter((p) => p.kind === "parameter")
          .map((p) => [p.id, p.default]),
      ),
    });
    check(
      (await workspace.request("card/open", { directory, id: added.id })).parameters[
        ids.INT
      ] === 20,
      "Workflow editor parameter changes were lost",
    );
    return {
      parameterTypes: 5,
      inheritedControls: true,
      reorder: true,
      workflowReload: true,
      projectReload: true,
      instanceIsolation: true,
      editRoundTrip: true,
    };
  });
  assert.deepEqual(errors, []);
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
}
