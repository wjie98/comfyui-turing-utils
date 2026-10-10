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
    () => window.LiteGraph?.registered_node_types?.TuringUtilsSeCTrackVisualConcept,
  );
  await page.waitForTimeout(1500);
  console.log(
    await page.evaluate(async () => {
      const { app } = await import("/scripts/app.js");
      if (LiteGraph.vueNodesMode) throw Error("Classic canvas required");
      app.graph.clear();
      let target = LiteGraph.createNode("TuringUtilsSeCTrackVisualConcept");
      app.graph.add(target);
      let source = LiteGraph.createNode("TuringUtilsMaskToVisualPrompts");
      app.graph.add(source);
      const check = (ok, message) => {
        if (!ok) throw Error(message);
      };
      const wait = () => new Promise((r) => setTimeout(r, 200));
      const positions = () => {
        const names = ["positive_coords", "negative_coords", "bounding_box"];
        const slots = names.map((name) =>
          target.inputs.findIndex((s) => s.name === name),
        );
        check(
          slots[0] >= 0 && slots[1] === slots[0] + 1 && slots[2] === slots[1] + 1,
          "Wrong coordinate/bbox input order",
        );
        return slots.map((i, index) => {
          const name = names[index];
          check(
            !target.widgets.some((w) => w.name === name),
            "Unexpected editor: " + name,
          );
          check(!target.inputs[i].widget, "Unexpected widget binding: " + name);
          return target.getConnectionPos(true, i)[1] - target.pos[1];
        });
      };
      await wait();
      const before = positions();
      for (const name of ["positive_coords", "negative_coords"]) {
        const output = source.outputs.findIndex((s) => s.name === name),
          input = target.inputs.findIndex((s) => s.name === name);
        check(source.connect(output, target, input), "Connection failed: " + name);
        await wait();
        check(
          JSON.stringify(positions()) === JSON.stringify(before),
          "Coordinate rows moved on connection",
        );
      }
      check(before[1] > before[0], "Overlapping coordinates");
      const prompt = await app.graphToPrompt();
      for (const name of ["positive_coords", "negative_coords"]) {
        const link = prompt.output[String(target.id)].inputs[name];
        check(
          Array.isArray(link) &&
            String(link[0]) === String(source.id) &&
            link[1] === source.outputs.findIndex((s) => s.name === name),
          "Local text replaced upstream connection: " + name,
        );
      }
      const saved = app.graph.serialize(),
        targetId = target.id,
        sourceId = source.id;
      await app.loadGraphData(saved);
      target = app.graph.getNodeById(targetId);
      source = app.graph.getNodeById(sourceId);
      await wait();
      positions();
      for (const name of ["positive_coords", "negative_coords"]) {
        const input = target.inputs.findIndex((s) => s.name === name);
        check(target.inputs[input].link != null, "Link lost during reload");
        target.disconnectInput(input);
        await wait();
        positions();
        check(
          source.connect(
            source.outputs.findIndex((s) => s.name === name),
            target,
            input,
          ),
          "Reconnect failed",
        );
      }
      return "SeC socket-only coordinates: order, independent connections, reload and reconnect OK";
    }),
  );
} finally {
  await browser.close();
}
