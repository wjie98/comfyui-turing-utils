import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
const { chromium } = await import(
  pathToFileURL(process.env.PLAYWRIGHT_MODULE).href
);
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH,
  args: ["--no-sandbox"],
});
const base = process.env.CANVAS_URL || "http://127.0.0.1:18188";
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(base);
  await page.waitForFunction(
    () =>
      window.app?.graph &&
      window.LiteGraph?.registered_node_types?.TuringCanvasInputs,
  );
  const result = await page.evaluate(async () => {
    const { app } = await import("/scripts/app.js");
    const { entries, syncPorts } =
      await import("/extensions/comfyui-turing-utils/lib/canvas_ports.js");
    const source = await (await fetch("/turing/workspace/new-template")).json();
    await app.loadGraphData(source);
    const first = await app.graphToPrompt();
    const assert = (yes, message) => {
      if (!yes) throw Error(message);
    };
    assert(
      Object.keys(first.output).length === 6,
      "starter must have six nodes",
    );
    const input = app.graph._nodes.find((n) => n.type === "TuringCanvasInputs");
    const output = app.graph._nodes.find(
      (n) => n.type === "TuringCanvasOutputs",
    );
    const text = app.graph._nodes.find((n) => n.type === "TuringMaterialText");
    assert(
      !text.widgets.some((w) =>
        ["prefix", "asset", "directory"].includes(w.name),
      ),
      "obsolete text fields",
    );
    const primitive = LiteGraph.createNode("PrimitiveString");
    app.graph.add(primitive);
    const dest = primitive.inputs.findIndex((i) => i.name === "value");
    const slot = input.outputs.length - 1;
    input.connect(slot, primitive, dest);
    assert(entries(input).length === 5, "dynamic input not appended");
    assert(entries(input).at(-1).type === "STRING", "dynamic type");
    const firstId = entries(input)[0].id;
    input.onMouseDown({ button: 0 }, [80, 24]);
    input.onMouseMove({}, [80, 80]);
    await new Promise(requestAnimationFrame);
    input.onMouseUp();
    assert(entries(input)[2].id === firstId, "label drag did not reorder");
    input.onMouseDown({ button: 0 }, [80, 24]);
    input.onMouseMove({}, [80, 108]);
    const beforeEscape = JSON.stringify(entries(input));
    input.onKeyDown({ key: "Escape" });
    assert(
      JSON.stringify(entries(input)) === beforeEscape,
      "Escape committed drag",
    );
    const prompt = app.canvas.prompt;
    app.canvas.prompt = (_, value, callback) => callback("Renamed input");
    try {
      input.onDblClick({}, [80, 24]);
    } finally {
      app.canvas.prompt = prompt;
    }
    assert(
      entries(input)[0].name === "Renamed input",
      "double-click rename failed",
    );
    primitive.connect(0, output, output.inputs.length - 1);
    assert(entries(output).length === 5, "dynamic output not appended");
    syncPorts(output, entries(output).slice(0, -1));
    const port = entries(input).at(-1);
    port.default = "inline";
    syncPorts(
      input,
      entries(input).map((p) => (p.id === port.id ? port : p)),
    );
    primitive.connect(
      0,
      text,
      text.inputs.findIndex((i) => i.name === "value"),
    );
    syncPorts(input, entries(input).reverse());
    syncPorts(output, entries(output).reverse());
    await app.loadGraphData(app.graph.serialize());
    const packed = await app.graphToPrompt();
    const r = await fetch("/turing/workspace/template/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: "basic-v2-" + Date.now() + ".json",
        workflow: packed.workflow,
        prompt: packed.output,
      }),
    });
    const saved = await r.json();
    assert(r.ok, JSON.stringify(saved));
    return {
      saved,
      starter: first,
      packed,
      textId: text.widgets.find((w) => w.name === "stub_id").value,
    };
  });
  const directory = "workspace-tests/basic-v2-" + Date.now();
  const post = async (path, data = {}) => {
    const r = await page.request.post(base + "/turing/workspace/" + path, {
      data: { directory, ...data },
    });
    const v = await r.json();
    assert(r.ok(), JSON.stringify(v));
    return v;
  };
  await post("project");
  const { id } = await post("card/add", { name: result.saved.name });
  const doc = await post("project");
  assert.equal(doc.cards.length, 1);
  const target = id + ":" + result.textId;
  const compiled = await post("compile", { target, revision: doc.revision });
  const queued = await page.request.post(base + "/prompt", {
    data: { prompt: compiled.prompt, prompt_id: compiled.run_id },
  });
  assert(queued.ok(), await queued.text());
  for (let i = 0; i < 100; i++) {
    const run = await post("run", { run_id: compiled.run_id });
    if (run.status === "success") break;
    await new Promise((r) => setTimeout(r, 100));
  }
  assert.equal((await post("project")).selections[target].text, "inline");
  await post("text", {
    node: target,
    text: "",
    revision: (await post("project")).selections[target].revision,
  });
  assert.equal((await post("project")).selections[target].text, "");
  const basic = await post("card/add", { kind: "text", x: 800, y: 0 });
  assert(basic.id);
  await page.goto(
    base + "/turing/workspace/?directory=" + encodeURIComponent(directory),
  );
  await page.waitForSelector("article.card");
  await page
    .getByRole("button", { name: "添加节点", exact: true })
    .dispatchEvent("click");
  await page.waitForSelector("dialog");
  assert.equal(
    await page
      .locator("dialog")
      .getByRole("button", { name: "图片", exact: true })
      .count(),
    1,
  );
  await page
    .locator("dialog")
    .getByRole("button", { name: "关闭", exact: true })
    .dispatchEvent("click");
  await page
    .getByRole("button", { name: "打开项目", exact: true })
    .dispatchEvent("click");
  await page.waitForSelector("dialog");
  assert.equal(await page.locator("dialog input").count(), 1);
  assert.deepEqual(errors, []);
  console.log(
    "Starter, dynamic endpoints, connected reorder/reload, text execution and empty text, basic cards, project/node pickers: OK",
  );
} finally {
  await browser.close();
}
