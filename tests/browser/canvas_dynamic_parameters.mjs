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
  page.on("console", (event) => {
    if (event.type() === "error") console.log("BROWSER ERROR", event.text());
    if (
      event.type() === "error" &&
      event.text().includes("Error calling extension 'TuringUtils")
    )
      errors.push(event.text());
  });
  await page.goto(process.env.CANVAS_URL || "http://127.0.0.1:18188");
  await page.waitForFunction(
    () =>
      window.app?.extensionManager.workflow.activeWorkflow?.isLoaded &&
      !window.app.extensionManager.workflow.isBusy,
  );
  const result = await page.evaluate(async () => {
    const { app } = await import("/scripts/app.js");
    const { api } = await import("/scripts/api.js");
    const workspace = await import(
      "/extensions/comfyui-turing-utils/material_workspace.js"
    );
    const { entries, syncPorts } = await import(
      "/extensions/comfyui-turing-utils/lib/canvas_ports.js"
    );
    const { DYNAMIC_COMBO, dynamicDefinition } = await import(
      "/extensions/comfyui-turing-utils/lib/dynamic_parameter.js"
    );
    const check = (condition, message) => {
      if (!condition) throw Error(message);
    };
    check(
      !LiteGraph.vueNodesMode,
      "This regression must run in the classic node editor",
    );
    const change = async (node, name, value) => {
      const w = node.widgets.find((w) => w.name === name);
      check(w, `Missing widget ${name}`);
      w.value = value;
      await w.callback(value);
    };
    const stamp = Date.now(),
      directory = `canvas-dynamic-${stamp}`,
      name = `canvas-dynamic-${stamp}.json`;
    await workspace.request("directory/create", { directory });
    await workspace.openProject(directory, true);
    await workspace.openTab(
      await (await fetch("/turing/workspace/new-template")).json(),
      name,
    );
    const endpoint = app.graph._nodes.find((n) => n.type === "TuringCanvasInputs");
    const image = app.graph._nodes.find((n) => n.type === "TuringMaterialImage");
    for (const node of [...app.graph._nodes])
      if (node.type.startsWith("TuringMaterial") && node !== image)
        app.graph.remove(node);
    syncPorts(
      endpoint,
      entries(endpoint).filter((p) => p.name === "Image"),
    );
    const outputs = app.graph._nodes.find((n) => n.type === "TuringCanvasOutputs");
    syncPorts(
      outputs,
      entries(outputs).filter((p) => p.type === "IMAGE"),
    );
    const outpaint = LiteGraph.createNode("TuringUtilsVideoPadForOutpaint");
    const attention = LiteGraph.createNode("TuringUtilsAttentionStrategy");
    const source = LiteGraph.createNode("EmptyImage");
    for (const n of [outpaint, attention, source]) app.graph.add(n);
    source.widgets.find((w) => w.name === "width").value = 64;
    source.widgets.find((w) => w.name === "height").value = 64;
    outpaint.widgets.find((w) => w.name === "megapixels").value = 0.01;
    source.connect(
      0,
      outpaint,
      outpaint.inputs.findIndex((s) => s.name === "images"),
    );
    outpaint.connect(
      0,
      image,
      image.inputs.findIndex((s) => s.name === "value"),
    );
    const ids = {};
    for (const [target, input] of [
      [outpaint, "layout"],
      [attention, "strategy"],
    ]) {
      check(
        endpoint.connect(
          endpoint.outputs.findIndex((s) => s._append),
          target,
          target.inputs.findIndex((s) => s.name === input),
        ),
        `Cannot expose ${input}`,
      );
      const port = entries(endpoint).at(-1);
      check(
        port.kind === "parameter" && port.type === DYNAMIC_COMBO,
        "Wrong group definition",
      );
      ids[input] = port.id;
    }
    const fieldName = (id, path = "") => `parameter:${id}${path ? "." + path : ""}`;
    await change(endpoint, fieldName(ids.layout, "left"), 64);
    await change(endpoint, fieldName(ids.layout), "relative_frame");
    check(
      !endpoint.widgets.some((w) => w.name === fieldName(ids.layout, "left")),
      "Inactive branch is visible",
    );
    await change(endpoint, fieldName(ids.layout, "expand_ratio"), 0.5);
    await change(endpoint, fieldName(ids.layout, "offset_x"), 0.1);
    await change(endpoint, fieldName(ids.layout), "pixels");
    check(
      endpoint.widgets.find((w) => w.name === fieldName(ids.layout, "left")).value ===
        64,
      "Pixel branch value was lost",
    );
    await change(endpoint, fieldName(ids.strategy, "routing_threshold"), 0.7);
    await change(endpoint, fieldName(ids.strategy, "prefix_policy"), "manual");
    await change(
      endpoint,
      fieldName(ids.strategy, "prefix_policy.manual_prefix_tokens"),
      128,
    );
    await change(endpoint, fieldName(ids.strategy), "sla");
    await change(endpoint, fieldName(ids.strategy, "sparsity_ratio"), 0.6);
    await change(endpoint, fieldName(ids.strategy), "sol");
    check(
      endpoint.widgets.find(
        (w) => w.name === fieldName(ids.strategy, "routing_threshold"),
      ).value === 0.7,
      "Sol value lost",
    );
    check(
      endpoint.widgets.find(
        (w) => w.name === fieldName(ids.strategy, "prefix_policy.manual_prefix_tokens"),
      ).value === 128,
      "Nested value lost",
    );
    check(
      !endpoint.inputs.some(
        (s) => s.type === DYNAMIC_COMBO || s.name.startsWith("parameter:"),
      ),
      "Group form leaked input sockets",
    );
    check(
      !endpoint.connect(
        entries(endpoint).find((p) => p.id === ids.layout).slot,
        attention,
        attention.inputs.findIndex((s) => s.name === "strategy"),
      ),
      "Mismatched schemas accepted",
    );
    check(
      dynamicDefinition(
        {
          constructor: {
            nodeData: {
              input: {
                required: {
                  test: [
                    DYNAMIC_COMBO,
                    {
                      options: [
                        { key: "bad", inputs: { required: { model: ["MODEL", {}] } } },
                      ],
                    },
                  ],
                },
              },
            },
          },
          widgets: [{ name: "test", value: "bad" }],
        },
        { name: "test" },
      ) === null,
      "MODEL branch accepted",
    );
    const reversed = entries(endpoint).reverse();
    reversed.find((p) => p.id === ids.layout).name = "Placement";
    syncPorts(endpoint, reversed);
    let snapshot = await app.graphToPrompt();
    const inputs = snapshot.output[String(attention.id)].inputs;
    check(
      inputs.strategy === "sol" &&
        inputs["strategy.prefix_policy.manual_prefix_tokens"] === 128,
      "Native API group expansion wrong",
    );
    check(
      !("strategy.sparsity_ratio" in inputs) &&
        !("strategy.prefix_policy.sparse_reference_image" in inputs),
      "Inactive API inputs remain",
    );
    check(
      snapshot.output[String(outpaint.id)]._meta.turing_dynamic_inputs.layout[1] ===
        entries(endpoint).find((p) => p.id === ids.layout).slot,
      "Reorder broke binding",
    );
    // A second line into an owned branch must not be silently overwritten.
    const number = LiteGraph.createNode("PrimitiveInt");
    app.graph.add(number);
    number.connect(
      0,
      outpaint,
      outpaint.inputs.findIndex((s) => s.name === "layout.left"),
    );
    let rejected = false;
    try {
      await app.graphToPrompt();
    } catch (e) {
      rejected = e.message.includes("Disconnect branch");
    }
    check(rejected, "Mixed group/leaf ownership did not fail");
    app.graph.remove(number);
    await workspace.openTab(snapshot.workflow, `dynamic-reload-${stamp}.json`);
    const restored = app.graph.getNodeById(endpoint.id);
    check(
      restored.widgets.find(
        (w) => w.name === fieldName(ids.strategy, "prefix_policy.manual_prefix_tokens"),
      ).value === 128,
      "Workflow reload lost nested state",
    );
    snapshot = await app.graphToPrompt();
    const queued = await api.queuePrompt(0, snapshot);
    let standalone = false;
    for (let i = 0; i < 80; i++) {
      const history = await (await api.fetchApi(`/history/${queued.prompt_id}`)).json();
      const task = history[queued.prompt_id];
      if (task?.status.completed) {
        check(
          task.status.status_str === "success",
          "Standalone workflow execution failed",
        );
        standalone = true;
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    check(standalone, "Standalone DynamicCombo workflow did not finish");
    await workspace.request("template/save", {
      name,
      workflow: snapshot.workflow,
      prompt: snapshot.output,
    });
    const template = await workspace.request("template/open", { name });
    await workspace.openProject(directory);
    const added = await workspace.addCard({ name });
    let card = app.graph._nodes.find((n) => n.properties.instance === added.id);
    check(
      card.properties.card.fields.length === 2,
      "Dynamic groups did not become form fields",
    );
    check(card.inputs.length === 0, "Card group leaked sockets");
    const w = card.widgets.find((w) => w.name === ids.layout);
    check(
      w.label === "Placement" && w.type === "combo",
      "Native dropdown label missing",
    );
    const nestedName = `${ids.strategy}.prefix_policy.manual_prefix_tokens`;
    check(
      card.widgets.find((w) => w.name === nestedName).value === 128,
      "Projected nested values wrong",
    );
    const selection = card.properties.materials.find((m) => m.kind === "image");
    const execute = async () => {
      const before = await workspace.request("selection", {
        directory,
        node: selection.id,
      });
      await card.widgets.find((w) => w.name === "image · 执行到这里").callback();
      for (let i = 0; i < 80; i++) {
        const current = await workspace.request("selection", {
          directory,
          node: selection.id,
        });
        if (current.asset && current.revision > (before.revision || 0)) return current;
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
      throw Error("Local dynamic execution did not publish image");
    };
    const first = await execute();
    await change(card, ids.layout, "relative_frame");
    check(
      card.widgets.find((w) => w.name === `${ids.layout}.expand_ratio`).value === 0.5,
      "Inactive branch not restored on card",
    );
    await change(card, `${ids.layout}.offset_x`, 0.2);
    const second = await execute();
    check(first.asset !== second.asset, "Second branch did not generate new material");
    // Repeated branch switches must not grow widgets or issue duplicate saves.
    await workspace.saveProject();
    const beforeRevision = (await workspace.request("project/open", { directory }))
      .document.revision;
    for (let i = 0; i < 6; i++) {
      await change(card, ids.layout, i % 2 ? "relative_frame" : "pixels");
      check(card.inputs.length === 0, "Repeated switch created sockets");
    }
    const after = await workspace.request("project/open", { directory });
    check(
      after.document.revision === beforeRevision + 6,
      "Switch callback saved more than once",
    );
    const opened = await workspace.request("card/open", { directory, id: added.id });
    check(
      opened.parameters[ids.layout].branches.relative_frame.offset_x === 0.2,
      "Group parameters not persisted",
    );
    check(
      opened.parameters[ids.layout].branches.pixels.left === 64,
      "Inactive values not persisted",
    );
    check(
      JSON.stringify((await workspace.request("template/open", { name })).workflow) ===
        JSON.stringify(template.workflow),
      "Instance edits changed template",
    );
    await workspace.openProject(directory);
    card = app.graph._nodes.find((n) => n.properties.instance === added.id);
    check(
      card.widgets.find((w) => w.name === `${ids.layout}.offset_x`)?.value === 0.2,
      "Project reopen lost values: " +
        JSON.stringify({
          state: card.properties.values[ids.layout],
          widgets: card.widgets
            .filter((w) => w.name.startsWith(ids.layout))
            .map((w) => [w.name, w.value]),
        }),
    );
    await card.widgets.find((w) => w.name === "编辑工作流").callback();
    const editing = app.graph.getNodeById(endpoint.id);
    check(
      editing.widgets.find((w) => w.name === fieldName(ids.layout, "offset_x"))
        .value === 0.2,
      "Edit instance lost values",
    );
    const edited = await app.graphToPrompt();
    await workspace.request("card/save", {
      directory,
      id: added.id,
      revision: opened.revision,
      workflow: edited.workflow,
      prompt: edited.output,
      parameters: Object.fromEntries(
        entries(editing)
          .filter((p) => p.kind === "parameter")
          .map((p) => [p.id, p.default]),
      ),
    });
    return {
      nativeDynamicCombo: true,
      nestedBranches: true,
      workflowReload: true,
      projectReload: true,
      executionBranches: 2,
      standaloneExecution: true,
      ownershipChecks: true,
      repeatedSwitches: 6,
      editRoundTrip: true,
    };
  });
  assert.deepEqual(errors, []);
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
}
