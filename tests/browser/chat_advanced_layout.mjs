import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
const { chromium } = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href);
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage();
  await page.goto(process.env.CANVAS_URL || "http://127.0.0.1:18188");
  await page.waitForFunction(
    () => window.LiteGraph?.registered_node_types?.TuringUtilsMultimodalPromptChat,
  );
  const result = await page.evaluate(async () => {
    const { app } = await import("/scripts/app.js");
    const { enableAdvancedLayout } = await import(
      "/extensions/comfyui-turing-utils/lib/advanced_layout.js"
    );
    if (LiteGraph.vueNodesMode) throw Error("Classic canvas required");
    const check = (ok, msg) => {
      if (!ok) throw Error(msg);
    };
    const tick = () =>
      new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    const type = "TuringUtilsMultimodalPromptChat";
    const def = (await (await fetch("/object_info/" + type)).json())[type];
    const refType = "NativeChatLayoutReference";
    await app.registerNodeDef(refType, { ...def, name: refType });
    app.graph.clear();
    let chat = LiteGraph.createNode(type),
      reference = LiteGraph.createNode(refType);
    app.graph.add(chat);
    app.graph.add(reference);
    await tick();
    const wrapped = chat.getLayoutWidgets;
    enableAdvancedLayout(chat);
    check(
      chat.getLayoutWidgets === wrapped,
      "Repeated installation wrapped layout again",
    );
    let cases = 0;
    async function audit(shown, width, height) {
      for (const n of [chat, reference]) {
        for (const w of n.widgets) w.advanced = !!w.options?.advanced;
        if (!!n.showAdvanced !== shown) n.toggleAdvanced();
      }
      // Native hidden-widget layout is the oracle; no plugin layout on reference.
      for (const w of reference.widgets) {
        w.advanced = !!w.options?.advanced;
        w.hidden = w.advanced && !shown;
      }
      chat.setSize([width, height]);
      reference.setSize([width, height]);
      await tick();
      for (const n of [chat, reference]) {
        n.arrange();
        n.arrange();
      }
      const a = chat.getLayoutWidgets(),
        b = reference.getLayoutWidgets();
      check(
        JSON.stringify(a.map((w) => w.name)) === JSON.stringify(b.map((w) => w.name)),
        "Hidden advanced controls still reserve layout space",
      );
      for (let i = 0; i < a.length; i++) {
        check(Math.abs(a[i].y - b[i].y) < 0.1, "Widget position differs: " + a[i].name);
        check(
          Math.abs(a[i].computedHeight - b[i].computedHeight) < 0.1,
          "Widget height differs: " + a[i].name,
        );
      }
      check(
        Math.abs(chat.freeWidgetSpace - reference.freeWidgetSpace) < 0.1,
        "Hidden controls consume resize space",
      );
      cases++;
    }
    const values = JSON.stringify(chat.serialize().widgets_values);
    for (const shown of [false, true, false])
      for (const [w, h] of [
        [360, 900],
        [720, 1200],
        [400, 600],
        [520, 1000],
      ])
        await audit(shown, w, h);
    check(
      JSON.stringify(chat.serialize().widgets_values) === values,
      "Resize/toggle changed values",
    );
    const id = chat.id,
      refId = reference.id;
    await app.loadGraphData(app.graph.serialize());
    chat = app.graph.getNodeById(id);
    reference = app.graph.getNodeById(refId);
    await tick();
    for (const shown of [true, false]) await audit(shown, 480, 1100);
    check(
      JSON.stringify(chat.serialize().widgets_values) === values,
      "Reload changed values",
    );
    for (const format of ["png", "jpeg"]) {
      for (const n of [chat, reference]) {
        const w = n.widgets.find((w) => w.name === "image_format");
        w.value = format;
        w.callback?.(format);
      }
      await tick();
      for (const shown of [false, true, false]) await audit(shown, 640, 1000);
    }
    return { cases, classic: true, nativeLayoutReference: true, reload: true };
  });
  assert.ok(result.cases >= 14);
  console.log("Chat folded layout:", result);
} finally {
  await browser.close();
}
