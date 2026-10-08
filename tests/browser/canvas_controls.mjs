// Run against a development server with Playwright installed externally.
import { pathToFileURL } from "node:url";
const {chromium} = await import(process.env.PLAYWRIGHT_MODULE ? pathToFileURL(process.env.PLAYWRIGHT_MODULE).href : "playwright");
const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_PATH, args: ["--no-sandbox"]});
const page = await browser.newPage({viewport: {width: 1800, height: 1200}});
const errors = [];
page.on("pageerror", error => errors.push(String(error)));
await page.route("**/turing/canvas/loras", route => route.fulfill({json: {names: ["alpha.safetensors", "folder/beta.safetensors"]}}));
await page.route("**/turing/canvas/history", route => route.fulfill({json: {items: [{id: "example.mp4", name: "example.mp4"}]}}));
try {
  await page.goto(process.env.CANVAS_URL || "http://127.0.0.1:18188");
  await page.waitForFunction(() => window.app?.graph && window.LiteGraph?.registered_node_types?.TuringCanvasH3Settings);
  await page.waitForTimeout(1500);
  console.log(await page.evaluate(async () => {
    const check = (ok, text) => { if (!ok) throw Error(text); };
    check(!LiteGraph.vueNodesMode, "This regression must run on the classic canvas");
    app.graph.clear();
    const add = type => { const node = LiteGraph.createNode(type); app.graph.add(node); return node; };
    const store = window.comfyAPI?.nodeDefStore?.useNodeDefStore?.() ?? app.extensionManager?._p?._s?.get("nodeDef");
    check(store?.nodeDefFilters?.some(f => f.id === "turing.internal"), "Internal search filter missing");
    for (const [name, type] of Object.entries(LiteGraph.registered_node_types)) {
      if (!name.startsWith("_Turing") && name !== "TuringUtilsStagePath") continue;
      type.skip_list = false;
      check(type.skip_list, "Internal node visible in menu: " + name);
      check(!store.visibleNodeDefs.some(def => def.name === name), "Internal node visible in search: " + name);
    }
    const publicTypes = Object.keys(LiteGraph.registered_node_types).filter(name => name.startsWith("TuringUtils") && name !== "TuringUtilsStagePath");
    let branches = 0;
    for (const type of publicTypes) {
    const sec = add(type);
    sec.showAdvanced = true;
    const serializedOrder = (sec.widgets ?? []).map(w => w.name);
    sec.showAdvanced = false;
    check(sec.getLayoutWidgets().every(w => !w.advanced), 'Classic canvas leaked advanced controls: ' + type);
    sec.showAdvanced = true;
    const layout = sec.getLayoutWidgets();
    let advanced = false;
    for (const w of layout) {
      check(!advanced || !!w.advanced, "SeC advanced controls precede ordinary controls");
      advanced ||= !!w.advanced;
    }
    check(JSON.stringify((sec.widgets ?? []).map(w => w.name)) === JSON.stringify(serializedOrder), "Saved order changed: " + type);
    const definition = LiteGraph.registered_node_types[type].nodeData;
    for (const [name, spec] of Object.entries({...definition.input?.required, ...definition.input?.optional})) {
      if (spec[0] !== "COMFY_DYNAMICCOMBO_V3") continue;
      for (const option of spec[1].options) {
        const widget = sec.widgets.find(w => w.name === name);
        widget.value = option.key;
        await widget.callback?.(option.key);
        await Promise.resolve();
        const visible = sec.getLayoutWidgets();
        const firstAdvanced = visible.findIndex(w => w.advanced);
        check(firstAdvanced < 0 || visible.slice(firstAdvanced).every(w => w.advanced), `Dynamic advanced ordering: ${type}/${name}/${option.key}`);
        const names = sec.widgets.map(w => w.name);
        if (type === "TuringUtilsMultimodalPromptChat" && name === "image_format") {
          const quality = sec.widgets.find(w => w.name === "image_format.jpeg_quality");
          check(option.key === "jpeg" ? quality?.advanced : !quality, "JPEG child visibility/advanced state incorrect");
        }
        check(new Set(names).size === names.length, `Duplicate dynamic widgets: ${type}`);
        check([...sec.computeSize()].every(Number.isFinite), `Invalid dynamic size: ${type}`);
        branches++;
      }
    }
    const savedNode = sec.serialize();
    const clone = add(type); clone.configure({...savedNode, id: clone.id});
    check(JSON.stringify(clone.serialize().widgets_values) === JSON.stringify(savedNode.widgets_values), `Widget roundtrip: ${type}`);
    check([...clone.computeSize()].every(Number.isFinite), `Invalid restored size: ${type}`);
    app.graph.remove(clone);
    app.graph.remove(sec);
    }
    console.log(`Audited ${publicTypes.length} public nodes and ${branches} dynamic branches`);
    const root = add("TuringCanvasSettings"), settings = add("TuringCanvasH3Settings");
    const assets = ["Image", "Video", "Audio", "H3"].map(type => add("TuringCanvas" + type));
    const storage = settings.widgets.find(w => w.name === "loras");
    storage.value = JSON.stringify([{name: "alpha.safetensors", on: true, strength: 1}, {name: "folder/beta.safetensors", on: false, strength: .5}]);
    storage.callback();
    const settingsLayout = settings.getLayoutWidgets();
    const addIndex = settingsLayout.findLastIndex(w => w.canvasLoraRow);
    check(settingsLayout[addIndex + 1]?.name === "shift_video" && settingsLayout[addIndex + 2]?.name === "shift_audio", "Shift controls not below LoRA");
    check(settings.widgets.filter(w => w.canvasLoraRow).length === 4, "LoRA row count");
    check(!settingsLayout.includes(storage), "Internal JSON control is visible");
    check(!settings.widgets.some(w => w.name === "lora_stack"), "HTML LoRA overlay remains");
    const sizes = [];
    for (const node of assets) {
      const min = node.computeSize();
      node.setSize([100000, 100000]);
      const large = [...node.size];
      check(large[0] <= 1600 && large[1] <= min[1] + 900, `Unbounded node size ${node.type} ${JSON.stringify({min, large})}`);
      for (let i = 0; i < 30; i++) node.setSize([...node.size]);
      check(node.size[1] === large[1], "Resize feedback loop");
      node.setSize([390, min[1]]);
      check(node.size[1] < large[1], "Cannot shrink preview");
      check(node.widgets.find(w => w.name === "material_preview").computeSize()[0] === 300, "Minimum depends on width");
      check(node.widgets.find(w => w.name === (node.type === "TuringCanvasH3" ? "历史生成结果" : "asset_id")).type === "combo", "History not native combo");
      const history = node.widgets.find(w => w.name === (node.type === "TuringCanvasH3" ? "历史生成结果" : "asset_id"));
      const nativeCombo = assets.at(-1).widgets.find(w => w.name === "历史生成结果");
      check(Object.getPrototypeOf(history) === Object.getPrototypeOf(nativeCombo), "History retained a text widget implementation");
      sizes.push([node.type, [...node.size], large]);
    }
    const saved = app.graph.serialize();
    await app.loadGraphData(saved);
    const loaded = app.graph._nodes.find(n => n.type === "TuringCanvasH3Settings");
    check(JSON.parse(loaded.widgets.find(w => w.name === "loras").value)[1].strength === .5, "LoRA roundtrip");
    loaded.pos = [200, 100]; app.canvas.ds.scale = 1; app.canvas.ds.offset = [0, 0];
    for (const n of app.graph._nodes) if (n !== loaded) n.pos = [2200, 0];
    app.graph.setDirtyCanvas(true, true);
    return {sizes, auditedNodes: publicTypes.length, dynamicBranches: branches};
  }));
  await page.waitForTimeout(800);
  const point = await page.evaluate(() => {
    const node = app.graph._nodes.find(n => n.type === "TuringCanvasH3Settings");
    const rows = node.widgets.filter(w => w.canvasLoraEntry);
    const following = node.widgets.find(w => w.name === "sampler_name");
    if (!(following.last_y >= rows.at(-1).last_y + 20)) throw Error(`Rows overlap following parameter ${following.last_y} ${rows.at(-1).last_y}`);
    const rect = app.canvas.canvas.getBoundingClientRect();
    return {x: rect.x + node.pos[0] + 55, y: rect.y + node.pos[1] + rows[0].last_y + 13};
  });
  // Exercise the widget contract directly; this fixture has no active workflow tab.
  await page.evaluate(({x, y}) => {
    app.canvas.ds.scale = 2.5;
    const node = app.graph._nodes.find(n => n.type === "TuringCanvasH3Settings");
    const row = node.widgets.find(w => w.canvasLoraEntry);
    row.mouse(new PointerEvent("pointerdown", {clientX:x, clientY:y}), [55, row.last_y + 13], node);
    row.mouse(new PointerEvent("pointerup", {clientX:x, clientY:y}), [55, row.last_y + 13], node);
  }, point);
  if (process.env.CANVAS_SCREENSHOT) await page.screenshot({path:process.env.CANVAS_SCREENSHOT});
  await page.waitForSelector(".litecontextmenu");
  console.log(await page.evaluate(() => {
    const menu = document.querySelector(".litecontextmenu"), rect = menu.getBoundingClientRect();
    if (menu.style.transform !== "none" || rect.right > innerWidth || rect.bottom > innerHeight) throw Error("Unbounded menu");
    return {menuBounds: "OK"};
  }));
  await page.locator(".litecontextmenu .litemenu-entry").filter({hasText: "folder/beta.safetensors"}).dispatchEvent("click");
  await page.waitForTimeout(300);
  console.log(await page.evaluate(() => {
    const node = app.graph._nodes.find(n => n.type === "TuringCanvasH3Settings");
    const storage = node.widgets.find(w => w.name === "loras");
    if (JSON.parse(storage.value)[0].name !== "folder/beta.safetensors") throw Error("Native LoRA selection failed: " + storage.value);
    const row = node.widgets.find(w => w.canvasLoraEntry);
    const click = part => {
      const b = row.hitAreas[part], pos = [b[0] + b[2]/2, b[1] + b[3]/2];
      row.mouse(new PointerEvent("pointerdown"), pos, node);
      row.mouse(new PointerEvent("pointerup"), pos, node);
    };
    click("toggle");
    if (JSON.parse(storage.value)[0].on !== false) throw Error("LoRA toggle failed");
    click("strengthInc");
    if (JSON.parse(storage.value)[0].strength !== 1.05) throw Error("LoRA strength failed");
    const prompt = app.canvas.prompt;
    try {
      for (const value of ["-2.345678", "12.000123"]) {
        app.canvas.prompt = (_title, _value, callback) => callback(value);
        click("strengthVal");
        if (JSON.parse(storage.value)[0].strength !== Number(value)) throw Error("LoRA float precision/range lost");
      }
    } finally { app.canvas.prompt = prompt; }
    const slot = node.getSlotInPosition(node.pos[0] + 60, node.pos[1] + row.last_y + 10);
    node.getSlotMenuOptions(slot).find(item => item.content === "Remove LoRA").callback();
    if (JSON.parse(storage.value).length !== 1) throw Error("LoRA removal failed");
    return {loraSelection: "OK"};
  }));
  await page.waitForTimeout(700);
  let requests = 0; page.on("request", r => { if (r.url().includes("/turing/canvas/")) requests++; });
  await page.waitForTimeout(1500);
  if (requests || errors.length) throw Error(JSON.stringify({idleRequests: requests, errors}));
  console.log("No idle requests or browser errors");
} finally { await browser.close(); }
