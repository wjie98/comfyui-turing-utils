import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import fs from "node:fs/promises";
import { setTimeout as delay } from "node:timers/promises";
const { chromium } = await import(
  pathToFileURL(process.env.PLAYWRIGHT_MODULE).href
);
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH,
  args: ["--no-sandbox"],
});
const base = process.env.CANVAS_URL || "http://127.0.0.1:18188";
const directory = `workspace-tests/files-${Date.now()}`,
  name = `smoke-${Date.now()}.json`;
const context = await browser.newContext(),
  page = await context.newPage(),
  errors = [];
page.on("pageerror", (e) => errors.push(e.message));
const request = async (path, body = {}) => {
  const r = await context.request.post(`${base}/turing/workspace/${path}`, {
    data: { directory, ...body },
  });
  const data = await r.json();
  assert.ok(r.ok(), JSON.stringify(data));
  return data;
};
async function poll(check) {
  const until = Date.now() + 30000;
  while (Date.now() < until) {
    const result = await check();
    if (result) return result;
    await delay(100);
  }
  throw Error("Timed out waiting for API state");
}
try {
  await page.goto(base);
  await page.waitForFunction(
    () =>
      window.app?.graph &&
      window.LiteGraph?.registered_node_types?.TuringCanvasInputs,
  );
  const pack = await page.evaluate(async () => {
    const { app } = await import("/scripts/app.js");
    if (LiteGraph.vueNodesMode) throw Error("Classic editor required");
    app.graph.clear();
    const { addPort, entries, syncPorts } =
      await import("/extensions/comfyui-turing-utils/lib/canvas_ports.js");
    const { bindMaterials } =
      await import("/extensions/comfyui-turing-utils/material_workspace.js");
    const source = LiteGraph.createNode("PrimitiveString"),
      stub = LiteGraph.createNode("TuringMaterialText");
    app.graph.add(source);
    app.graph.add(stub);
    source.connect(
      0,
      stub,
      stub.inputs.findIndex((i) => i.name === "value"),
    );
    bindMaterials();
    await new Promise(requestAnimationFrame);
    const input = app.graph._nodes.find((n) => n.type === "TuringCanvasInputs");
    const parameter = addPort(input, "可调文字", "STRING");
    const ps = entries(input);
    ps.find((p) => p.id === parameter.id).default = "hello material";
    syncPorts(input, ps);
    input.connect(
      parameter.slot,
      source,
      source.inputs.findIndex((i) => i.name === "value"),
    );
    syncPorts(input, entries(input).reverse());
    const before = (await app.graphToPrompt()).output;
    if (before[stub.id].inputs.position[1] !== 1)
      throw Error("Position link did not follow reordered port");
    const first = app.graph.convertToSubgraph(new Set([source, stub]));
    await new Promise(requestAnimationFrame);
    const nestedStub = first.subgraph.nodes.find(
      (n) => n.type === "TuringMaterialText",
    );
    first.subgraph.convertToSubgraph(new Set([nestedStub]));
    await new Promise(requestAnimationFrame);
    // Exercise persisted native nested workflows, not only transient editor state.
    await app.loadGraphData(app.graph.serialize());
    const exported = await app.graphToPrompt();
    const material = Object.values(exported.output).find(
      (n) => n.class_type === "TuringMaterialText",
    );
    if (!material.inputs.position)
      throw Error("Nested link missing: " + JSON.stringify(exported.output));
    return {
      workflow: exported.workflow,
      prompt: exported.output,
      stub: material.inputs.stub_id,
      parameter: parameter.id,
    };
  });
  await request("template/save", {
    name,
    workflow: pack.workflow,
    prompt: pack.prompt,
  });
  const a = (await request("card/add", { name })).id,
    b = (await request("card/add", { name })).id;
  let project = await request("project");
  await request("patch", {
    revision: project.revision,
    changes: [
      {
        type: "input",
        id: a + ":parameters",
        name: pack.parameter,
        value: "only A",
      },
    ],
  });
  project = await request("project");
  assert.equal(
    project.prompt[b + ":parameters"].inputs[pack.parameter],
    "hello material",
  );
  await page.goto(
    `${base}/turing/workspace/?directory=${encodeURIComponent(directory)}`,
  );
  const card = page.locator(`[data-card="${a}"]`);
  await card.getByRole("button", { name: "执行到这里" }).dispatchEvent("click");
  await page.waitForFunction(
    ({ id }) =>
      document.querySelector(`[data-card="${id}"] textarea`)?.value ===
      "only A",
    { id: a },
  );
  const first = (await request("project")).selections[a + ":" + pack.stub]
    .asset;
  await card.getByRole("button", { name: "执行到这里" }).dispatchEvent("click");
  await poll(
    async () =>
      (await request("project")).selections[a + ":" + pack.stub]?.asset !==
      first,
  );
  await card.getByRole("button", { name: "执行到这里" }).waitFor();
  await card.locator("textarea").fill("manual");
  await card.locator("textarea").blur();
  await page.waitForFunction(
    () => document.querySelector("#status").textContent === "文本已保存",
  );
  await page.reload();
  await page.waitForFunction(
    ({ id }) =>
      document.querySelector(`[data-card="${id}"] textarea`)?.value ===
      "manual",
    { id: a },
  );
  const editor = await context.newPage();
  editor.on("dialog", (d) => d.accept());
  await editor.goto(
    `${base}/?turing_workspace=${encodeURIComponent(directory)}&card=${a}`,
  );
  await editor.waitForFunction(() => {
    const input = window.app?.graph?.nodes.find(
      (n) => n.type === "TuringCanvasInputs",
    );
    return (
      input &&
      JSON.parse(input.widgets.find((w) => w.name === "ports").value).some(
        (p) => p.default === "only A",
      )
    );
  });
  const back = await editor.evaluate(async () => {
    const { app } = await import("/scripts/app.js");
    const p = await app.graphToPrompt(),
      input = Object.values(p.output).find(
        (n) => n.class_type === "TuringCanvasInputs",
      );
    if (
      !input ||
      !JSON.parse(input.inputs.ports).some((p) => p.default === "only A")
    )
      return false;
    const stub = Object.values(p.output).find(
      (n) => n.class_type === "TuringMaterialText",
    );
    if (!stub.inputs.asset)
      throw Error("Selected material was not restored into editor");
    const endpoint = app.graph.nodes.find(
      (n) => n.type === "TuringCanvasInputs",
    );
    const { entries, syncPorts } =
      await import("/extensions/comfyui-turing-utils/lib/canvas_ports.js");
    const ps = entries(endpoint);
    ps.find((p) => p.kind === "value").default = "saved back";
    syncPorts(endpoint, ps);
    return await app.graphToPrompt();
  });
  const opened = await request("card/open", { id: a });
  await request("card/save", {
    id: a,
    revision: opened.revision,
    workflow: back.workflow,
    prompt: back.output,
  });
  project = await request("project");
  assert.equal(
    project.prompt[a + ":parameters"].inputs[pack.parameter],
    "saved back",
  );
  assert.equal(
    project.prompt[b + ":parameters"].inputs[pack.parameter],
    "hello material",
  );
  assert.equal(
    project.selections[a + ":" + pack.stub].asset,
    opened.selections[a + ":" + pack.stub].asset,
  );
  await editor.close();
  if (process.env.MATERIAL_VIDEO_FIXTURE) {
    const native = await context.newPage();
    await native.goto(base);
    await native.waitForFunction(
      () =>
        window.app?.graph &&
        window.LiteGraph?.registered_node_types?.TuringMaterialVideo,
    );
    const videoPack = await native.evaluate(async () => {
      const { app } = await import("/scripts/app.js");
      app.graph.clear();
      const n = LiteGraph.createNode("TuringMaterialVideo");
      app.graph.add(n);
      const { bindMaterials } =
        await import("/extensions/comfyui-turing-utils/material_workspace.js");
      bindMaterials();
      return await app.graphToPrompt();
    });
    const videoName = "video-" + name;
    await request("template/save", {
      name: videoName,
      workflow: videoPack.workflow,
      prompt: videoPack.output,
    });
    const v = (await request("card/add", { name: videoName })).id;
    const upload = await context.request.post(
      `${base}/turing/workspace/import?directory=${encodeURIComponent(directory)}&kind=video`,
      {
        multipart: {
          file: {
            name: "fixture.webm",
            mimeType: "video/webm",
            buffer: await fs.readFile(process.env.MATERIAL_VIDEO_FIXTURE),
          },
        },
      },
    );
    assert.ok(upload.ok(), await upload.text());
    const asset = (await upload.json()).asset;
    project = await request("project");
    const material = project.cards.find((c) => c.id === v).materials[0];
    await request("select", {
      node: material,
      selection: { asset },
      revision: 0,
    });
    await request("patch", {
      revision: project.revision,
      changes: [{ type: "position", id: v, x: 0, y: 0 }],
    });
    let streams = 0;
    page.on("request", (r) => {
      const u = new URL(r.url());
      if (
        u.pathname.endsWith("/asset") &&
        u.searchParams.get("asset") === asset &&
        !u.searchParams.has("thumbnail")
      )
        streams++;
    });
    await page.reload();
    const vc = page.locator(`[data-card="${v}"]`);
    await vc.locator("img").waitFor();
    assert.equal(streams, 0);
    assert.equal(await vc.locator("video").count(), 0);
    await vc.getByRole("button", { name: "▶ 播放" }).dispatchEvent("click");
    await page.waitForFunction(
      () => document.querySelector("video")?.readyState >= 2,
    );
    assert.ok(streams > 0);
    const range = await context.request.get(
      `${base}/turing/workspace/asset?directory=${encodeURIComponent(directory)}&asset=${encodeURIComponent(asset)}`,
      { headers: { Range: "bytes=0-99" } },
    );
    assert.equal(range.status(), 206);
    const many = { ...project, prompt: {}, selections: {}, cards: [] };
    for (let i = 0; i < 1000; i++) {
      const id = `stress-${i}`,
        stub = `${id}:material`;
      many.prompt[stub] = project.prompt[material];
      many.selections[stub] = { asset, revision: 1 };
      many.cards.push({
        id,
        title: id,
        x: (i % 25) * 420,
        y: Math.floor(i / 25) * 560,
        materials: [stub],
        fields: [],
        layout: [{ material: stub }],
        ports: [],
        outputs: [],
        instance: true,
      });
    }
    await page.route("**/turing/workspace/project", (route) =>
      route.fulfill({ json: many }),
    );
    streams = 0;
    await page.reload();
    await page.locator("article.card").first().waitFor();
    assert.ok((await page.locator("article.card").count()) < 30);
    assert.equal(await page.locator("video,audio").count(), 0);
    assert.equal(streams, 0);
    await page.unroute("**/turing/workspace/project");
    const h3 = await native.evaluate(async () => {
      const { app } = await import("/scripts/app.js");
      const { bindMaterials } =
        await import("/extensions/comfyui-turing-utils/material_workspace.js");
      const { entries, syncPorts, addPort } =
        await import("/extensions/comfyui-turing-utils/lib/canvas_ports.js");
      app.graph.clear();
      const a = LiteGraph.createNode("TuringMaterialText"),
        b = LiteGraph.createNode("TuringMaterialText");
      app.graph.add(a);
      app.graph.add(b);
      bindMaterials();
      const out = app.graph.nodes.find((n) => n.type === "TuringCanvasOutputs");
      const before = (await app.graphToPrompt()).output[out.id].inputs;
      const old = entries(out);
      syncPorts(out, [...old].reverse());
      const after = (await app.graphToPrompt()).output[out.id].inputs;
      if (
        JSON.stringify(before.port_0) !== JSON.stringify(after.port_1) ||
        JSON.stringify(before.port_1) !== JSON.stringify(after.port_0)
      )
        throw Error("Output input links lost on reorder");
      const input = app.graph.nodes.find(
        (n) => n.type === "TuringCanvasInputs",
      );
      const p1 = addPort(input, "A", "STRING"),
        p2 = addPort(input, "B", "STRING");
      a.connect(
        0,
        input,
        input.inputs.findIndex((s) => s.name === `port_${p1.slot}`),
      );
      b.connect(
        0,
        input,
        input.inputs.findIndex((s) => s.name === `port_${p2.slot}`),
      );
      syncPorts(input, entries(input).reverse());
      const incoming = (await app.graphToPrompt()).output[input.id].inputs;
      if (
        incoming.port_0?.[0] !== String(b.id) ||
        incoming.port_1?.[0] !== String(a.id)
      )
        throw Error("Input test links lost on reorder");
      await app.loadGraphData(app.graph.serialize());
      const restored = (await app.graphToPrompt()).output[input.id].inputs;
      if (JSON.stringify(incoming) !== JSON.stringify(restored))
        throw Error("Endpoint reload changed links");
      await app.loadGraphData(await (await fetch('/turing/workspace/h3-template')).json());
      bindMaterials();
      return await app.graphToPrompt();
    });
    await request("template/save", {
      name: "h3-" + name,
      workflow: h3.workflow,
      prompt: h3.output,
    });
    await native.close();
  }
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      directory,
      name,
      classic: true,
      nestedSubgraphs: 2,
      portReorder: true,
      instanceIsolation: true,
      repeatedExecution: true,
      textPersistence: true,
      editorRoundTrip: true,
      videoAndThousandCards: Boolean(process.env.MATERIAL_VIDEO_FIXTURE),
      h3Template: Boolean(process.env.MATERIAL_VIDEO_FIXTURE),
    }),
  );
} catch (error) {
  console.error({
    errors,
    status: await page
      .locator("#status")
      .textContent({ timeout: 1000 })
      .catch(() => null),
  });
  throw error;
} finally {
  await browser.close();
}
