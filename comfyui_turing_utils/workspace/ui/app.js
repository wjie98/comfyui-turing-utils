const $ = (id) => document.getElementById(id);
const viewport = $("viewport"),
  world = $("world"),
  canvas = $("wires");
const apiBase = new URL("./", location.href),
  comfyBase = new URL("../../", apiBase);
const user = new URL(location.href).searchParams.get("user") || "";
const userHeaders = user ? { "Comfy-User": user } : {};
let directory = "",
  documentState,
  epoch = 0,
  view = { x: 40, y: 40, z: 1 },
  frame = 0,
  saving = Promise.resolve();
let player = null,
  linking = null,
  dragging = null;
const mounted = new Map(),
  cells = new Map(),
  materialCards = new Map(),
  jobs = new Map();
const drafts = new Map();
const dependencies = new Map();
const sizes = new ResizeObserver((entries) => {
  let changed = false;
  for (const entry of entries) {
    const mountedCard = mounted.get(entry.target.dataset.card);
    if (!mountedCard || entry.target.classList.contains("compact")) continue;
    const height = Math.ceil(
      entry.borderBoxSize?.[0]?.blockSize || entry.contentRect.height,
    );
    if (height > 0 && mountedCard.card.measuredHeight !== height) {
      mountedCard.card.measuredHeight = height;
      changed = true;
    }
  }
  if (changed) {
    index();
    schedule();
  }
});
function materialSources(id) {
  if (dependencies.has(id)) return dependencies.get(id);
  const seen = new Set(),
    sources = new Set(),
    pending = [id];
  while (pending.length) {
    const key = pending.pop();
    if (seen.has(key)) continue;
    seen.add(key);
    if (key !== id && materialCards.has(key)) {
      sources.add(key);
      continue;
    }
    for (const value of Object.values(documentState.prompt[key]?.inputs || {}))
      if (
        Array.isArray(value) &&
        value.length === 2 &&
        typeof value[0] === "string"
      )
        pending.push(value[0]);
  }
  dependencies.set(id, sources);
  return sources;
}
let textSaving = Promise.resolve();
let draftTimer;
function flushText(id) {
  textSaving = textSaving
    .catch(() => {})
    .then(async () => {
      const draft = drafts.get(id);
      if (!draft) return;
      const result = await request("text", {
        node: id,
        revision: current(id).revision,
        text: draft.value,
        prefix: draft.prefix,
      });
      documentState.selections[id] = {
        asset: result.asset,
        revision: result.revision,
      };
      if (drafts.get(id) === draft) drafts.delete(id);
      status("文本已保存");
    });
  return textSaving;
}
async function flushDrafts() {
  clearTimeout(draftTimer);
  for (const id of drafts.keys()) await flushText(id);
}
window.addEventListener("beforeunload", (event) => {
  if (drafts.size) {
    event.preventDefault();
    event.returnValue = "";
  }
});
const clientId = crypto.randomUUID();
const protocolResponse = await fetch(new URL("protocol", apiBase), {headers: userHeaders});
if (!protocolResponse.ok) throw Error("Cannot load material protocol");
const kinds = (await protocolResponse.json()).materials;

function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}
function report(error) {
  status(error.message || String(error), true);
}
async function request(path, data) {
  const response = await fetch(new URL(path, apiBase), {
    method: "POST",
    headers: { ...userHeaders, "Content-Type": "application/json" },
    body: JSON.stringify({ directory, ...data }),
  });
  const result = await response.json();
  if (!response.ok) throw Error(result.error || JSON.stringify(result));
  return result;
}
function element(tag, text, cls) {
  const el = document.createElement(tag);
  if (text != null) el.textContent = text;
  if (cls) el.className = cls;
  return el;
}
function button(text, fn, cls) {
  const el = element("button", text, cls);
  el.onclick = () => Promise.resolve().then(fn).catch(report);
  return el;
}
function current(id) {
  return documentState.selections[id] || { revision: 0 };
}
function assetURL(asset, thumbnail = false, start = 0) {
  const url = new URL("asset", apiBase);
  url.search = new URLSearchParams({
    directory,
    asset,
    ...(thumbnail ? { thumbnail: "1", start } : {}),
  });
  return url.href;
}
function stopPlayer() {
  if (!player) return;
  player.media.pause();
  player.media.removeAttribute("src");
  player.media.load();
  player.media.remove();
  player.play.hidden = false;
  player = null;
}
function schedule() {
  if (!frame) frame = requestAnimationFrame(render);
}
function height(card) {
  if (card.measuredHeight) return card.measuredHeight;
  return (
    100 +
    (card.ports?.length || 0) * 38 +
    (card.outputs?.length || 0) * 38 +
    (card.fields?.length || 0) * 38 +
    card.materials.reduce(
      (n, id) =>
        n +
        (kinds[documentState.prompt[id]?.class_type] === "text" ? 290 : 350),
      0,
    )
  );
}
function index() {
  cells.clear();
  materialCards.clear();
  for (const card of documentState.cards) {
    card.height = height(card);
    for (const id of card.materials) materialCards.set(id, card);
    for (
      let x = Math.floor(card.x / 1024);
      x <= Math.floor((card.x + 370) / 1024);
      x++
    )
      for (
        let y = Math.floor(card.y / 1024);
        y <= Math.floor((card.y + card.height) / 1024);
        y++
      ) {
        const key = `${x},${y}`;
        if (!cells.has(key)) cells.set(key, new Set());
        cells.get(key).add(card);
      }
  }
}
function visible() {
  const bounds = {
      x: -view.x / view.z - 400,
      y: -view.y / view.z - 400,
      r: (viewport.clientWidth - view.x) / view.z + 400,
      b: (viewport.clientHeight - view.y) / view.z + 400,
    },
    found = new Set();
  for (
    let x = Math.floor(bounds.x / 1024);
    x <= Math.floor(bounds.r / 1024);
    x++
  )
    for (
      let y = Math.floor(bounds.y / 1024);
      y <= Math.floor(bounds.b / 1024);
      y++
    )
      for (const card of cells.get(`${x},${y}`) || [])
        if (
          card.x + 370 >= bounds.x &&
          card.x <= bounds.r &&
          card.y + card.height >= bounds.y &&
          card.y <= bounds.b
        )
          found.add(card);
  if (dragging?.card) found.add(dragging.card);
  return found;
}
function render() {
  frame = 0;
  if (!documentState) return;
  world.style.transform = `translate(${view.x}px,${view.y}px) scale(${view.z})`;
  const shown = visible();
  for (const [id, entry] of mounted)
    if (!shown.has(entry.card)) {
      if (player?.card === id) stopPlayer();
      sizes.unobserve(entry.root);
      entry.root.remove();
      mounted.delete(id);
    }
  for (const card of shown) {
    let entry = mounted.get(card.id);
    const compact = view.z < 0.3;
    if (entry && entry.compact !== compact) {
      if (player?.card === card.id) stopPlayer();
      sizes.unobserve(entry.root);
      entry.root.remove();
      mounted.delete(card.id);
      entry = null;
    }
    if (!entry) {
      entry = { card, root: buildCard(card, compact), compact };
      mounted.set(card.id, entry);
      world.append(entry.root);
      sizes.observe(entry.root);
    }
    entry.root.style.left = card.x + "px";
    entry.root.style.top = card.y + "px";
    entry.root.classList.toggle("compact", view.z < 0.3);
  }
  drawWires(shown);
  $("empty").hidden = documentState.cards.length > 0;
}
function drawWires(shown) {
  const dpr = devicePixelRatio || 1,
    w = viewport.clientWidth,
    h = viewport.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.translate(view.x, view.y);
  ctx.scale(view.z, view.z);
  ctx.strokeStyle = "#657a93";
  ctx.lineWidth = 2 / view.z;
  for (const card of shown)
    for (const id of card.materials) {
      for (const key of materialSources(id)) {
        const source = materialCards.get(key);
        if (!source || source === card) continue;
        ctx.beginPath();
        ctx.moveTo(source.x + 370, source.y + 25);
        ctx.bezierCurveTo(
          source.x + 430,
          source.y + 25,
          card.x - 60,
          card.y + 25,
          card.x,
          card.y + 25,
        );
        ctx.stroke();
      }
    }
}
function patch(changes) {
  const token = epoch;
  saving = saving
    .catch(() => {})
    .then(async () => {
      if (token !== epoch) return;
      const result = await request("patch", {
        revision: documentState.revision,
        changes,
      });
      if (token === epoch) documentState.revision = result.revision;
    });
  return saving;
}
async function open() {
  await flushDrafts();
  await saving.catch(() => {});
  epoch++;
  dependencies.clear();
  stopPlayer();
  sizes.disconnect();
  for (const entry of mounted.values()) entry.root.remove();
  mounted.clear();
  directory = $("directory").value;
  documentState = await request("project", {});
  index();
  schedule();
  status("项目已打开");
}
function redrawMaterial(id) {
  const card = materialCards.get(id);
  if (!card) return;
  const entry = mounted.get(card.id);
  if (entry) {
    if (player?.card === card.id) stopPlayer();
    sizes.unobserve(entry.root);
    entry.root.remove();
    mounted.delete(card.id);
  }
  index();
  schedule();
}
async function select(id, selection) {
  const token = epoch;
  const result = await request("select", {
    node: id,
    revision: current(id).revision,
    selection,
  });
  if (token !== epoch) return;
  documentState.selections[id] = { ...selection, revision: result.revision };
  redrawMaterial(id);
}
function buildCard(card, compact = false) {
  const root = element("article", null, "card");
  root.dataset.card = card.id;
  const heading = element("h3", card.title);
  heading.onpointerdown = (e) => {
    if (e.button !== 0) return;
    heading.setPointerCapture(e.pointerId);
    dragging = {
      card,
      startX: e.clientX,
      startY: e.clientY,
      x: card.x,
      y: card.y,
    };
  };
  heading.onpointermove = (e) => {
    if (!dragging || dragging.card !== card) return;
    card.x = dragging.x + (e.clientX - dragging.startX) / view.z;
    card.y = dragging.y + (e.clientY - dragging.startY) / view.z;
    schedule();
  };
  heading.onpointerup = () => {
    if (dragging?.card !== card) return;
    dragging = null;
    index();
    patch([{ type: "position", id: card.id, x: card.x, y: card.y }]).catch(
      report,
    );
  };
  root.append(heading);
  if (compact) return root;
  if (card.error) root.append(element("p", card.error, "error"));
  if (card.instance) {
    root.append(
      button("编辑工作流", async () => {
        await flushDrafts();
        await saving;
        const url = new URL(comfyBase);
        url.searchParams.set("turing_workspace", directory);
        url.searchParams.set("card", card.id);
        window.open(url, "_blank", "noopener");
      }),
    );
    for (const port of card.ports || []) {
      const row = element("div", null, "row");
      row.append(element("span", `${port.name} (${port.type})`));
      row.append(
        button(card.bindings?.[port.id] ? "更换输入" : "连接输入", async () => {
          if (!linking) throw Error("先点击来源素材的输出圆点");
          await saving;
          await request("card/connect", {
            revision: documentState.revision,
            target: card.id,
            port: port.id,
            source: linking.source,
            source_slot: linking.source_slot,
          });
          linking = null;
          await open();
        }),
      );
      if (card.bindings?.[port.id])
        row.append(
          button("断开", async () => {
            await saving;
            await request("card/connect", {
              revision: documentState.revision,
              target: card.id,
              port: port.id,
              source: null,
            });
            await open();
          }),
        );
      root.append(row);
    }
  }
  const layout = card.layout || [
    ...(card.fields || []).map((field) => ({ field })),
    ...card.materials.map((material) => ({ material })),
  ];
  for (const item of layout) {
    if (item.material) {
      root.append(buildMaterial(item.material, card));
      continue;
    }
    const field = item.field;
    const node = documentState.prompt[field.node];
    if (!node || Array.isArray(node.inputs[field.input])) continue;
    const row = element("div", null, "field"),
      label = element("label", field.label || field.input),
      value = node.inputs[field.input];
    let input;
    if (field.options?.length) {
      input = element("select");
      for (const option of field.options)
        input.append(new Option(String(option), String(option)));
      input.value = String(value);
    } else if (typeof value === "boolean") {
      input = element("input");
      input.type = "checkbox";
      input.checked = value;
    } else {
      input = element("input");
      input.type = typeof value === "number" ? "number" : "text";
      input.value = value ?? "";
      if (input.type === "number") input.step = "any";
    }
    input.onchange = () => {
      node.inputs[field.input] =
        typeof value === "boolean"
          ? input.checked
          : typeof value === "number"
            ? Number(input.value)
            : input.value;
      patch([
        {
          type: "input",
          id: field.node,
          name: field.input,
          value: node.inputs[field.input],
        },
      ]).catch(report);
    };
    input.disabled = Boolean(card.bindings?.[field.input]);
    if (field.connected)
      input.title = "当前由已连接的输入提供；修改此值会覆盖内部测试默认输入";
    if (field.type === "INT") input.step = "1";
    row.append(label, input);
    root.append(row);
  }
  for (const port of card.outputs || []) {
    const row = element("div", null, "row");
    row.append(
      element("span", `${port.name} (${port.type})`),
      button(
        "",
        () => {
          linking = port;
          status("点击目标卡片端点的“连接输入”");
        },
        "port out",
      ),
    );
    root.append(row);
  }
  return root;
}
function buildMaterial(id, card) {
  const node = documentState.prompt[id],
    kind = kinds[node.class_type],
    selection = current(id),
    token = epoch;
  const box = element("section", null, `material ${kind}`);
  box.dataset.node = id;
  const heading = element("h4", null);
  heading.append(element("span", node._meta?.title || kind));
  box.append(heading);
  if (kind === "text") {
    const text = element("textarea");
    text.placeholder = "输入文本，离开编辑框时自动保存";
    box.append(text);
    let dirty = drafts.has(id);
    if (dirty) text.value = drafts.get(id).value;
    text.oninput = () => {
      dirty = true;
      drafts.set(id, {
        value: text.value,
        prefix: node.inputs.prefix || "text",
      });
      clearTimeout(draftTimer);
      draftTimer = setTimeout(() => flushDrafts().catch(report), 400);
    };
    text.onchange = () => flushText(id).catch(report);
    if (selection.asset)
      fetch(assetURL(selection.asset))
        .then((r) => {
          if (!r.ok) throw Error("无法读取文本");
          return r.text();
        })
        .then((value) => {
          if (token === epoch && text.isConnected && !dirty) text.value = value;
        })
        .catch(report);
  } else {
    const preview = element("div", null, "preview");
    box.append(preview);
    if (selection.asset) {
      if (kind !== "audio") {
        const image = element("img");
        image.src = assetURL(selection.asset, true, selection.start || 0);
        image.decoding = "async";
        image.loading = "lazy";
        image.alt = "素材预览";
        preview.append(image);
      }
      if (kind !== "image") {
        const play = button(
          "▶ 播放",
          async () => {
            stopPlayer();
            const media = element(kind);
            media.controls = true;
            media.loop = false;
            media.preload = "none";
            media.src = assetURL(selection.asset);
            preview.append(media);
            play.hidden = true;
            player = { media, play, card: card.id };
            media.onloadedmetadata = () => {
              media.currentTime = selection.start || 0;
            };
            media.onplay = () => {
              if (
                media.currentTime < (selection.start || 0) ||
                (selection.end && media.currentTime >= selection.end)
              )
                media.currentTime = selection.start || 0;
            };
            media.onseeking = () => {
              if (media.currentTime < (selection.start || 0))
                media.currentTime = selection.start || 0;
              if (selection.end && media.currentTime > selection.end) {
                media.currentTime = selection.end;
                media.pause();
              }
            };
            media.ontimeupdate = () => {
              if (selection.end && media.currentTime >= selection.end)
                media.pause();
            };
            try {
              await media.play();
            } catch (error) {
              stopPlayer();
              throw error;
            }
          },
          "play",
        );
        preview.append(play);
      }
    } else preview.append(element("span", "拖入素材或选择历史"));
    const upload = element("input");
    upload.type = "file";
    upload.accept = kind + "/*";
    upload.hidden = true;
    upload.onchange = () => importFile(id, kind, upload.files[0]).catch(report);
    box.append(upload);
    box.append(button("导入素材", () => upload.click()));
    box.ondragover = (e) => e.preventDefault();
    box.ondrop = (e) => {
      e.preventDefault();
      if (e.dataTransfer.files[0])
        importFile(id, kind, e.dataTransfer.files[0]).catch(report);
    };
  }
  const row = element("div", null, "row"),
    history = element("select");
  history.setAttribute("aria-label", "历史素材");
  history.append(
    new Option(
      selection.asset?.split("/").pop() || "选择历史素材",
      selection.asset || "",
    ),
  );
  let offset = 0,
    loaded = false;
  const loadHistory = async () => {
    if (loaded) return;
    loaded = true;
    const data = await request("history", {
      kind,
      offset,
      prefix: Array.isArray(node.inputs.value)
        ? node.inputs.prefix || kind
        : "",
    });
    if (token !== epoch || !history.isConnected) return;
    for (const item of data.items)
      if (![...history.options].some((o) => o.value === item.id))
        history.append(new Option(item.name, item.id));
    offset += data.items.length;
    if (data.items.length === 50)
      history.append(new Option("加载更多…", "__more"));
  };
  history.onpointerdown = () => loadHistory().catch(report);
  history.onfocus = () => loadHistory().catch(report);
  history.onchange = async () => {
    try {
      if (history.value === "__more") {
        history.querySelector('option[value="__more"]').remove();
        loaded = false;
        await loadHistory();
        return;
      }
      if (history.value) await select(id, { asset: history.value });
    } catch (error) {
      report(error);
    }
  };
  row.append(history);
  box.append(row);
  if (kind === "video" || kind === "audio") {
    const range = element("div", null, "row");
    for (const [key, label] of [
      ["start", "起点秒"],
      ["end", "终点秒（0=末尾）"],
    ]) {
      const wrap = element("label", label),
        control = element("input");
      control.type = "number";
      control.min = 0;
      control.step = 0.01;
      control.value = selection[key] || 0;
      control.onchange = () =>
        select(id, { ...current(id), [key]: Number(control.value) }).catch(
          report,
        );
      wrap.append(control);
      range.append(wrap);
    }
    box.append(range);
  }
  if (Array.isArray(node.inputs.value)) {
    const run = button("执行到这里", () => execute(id), "run");
    run.disabled = jobs.has(id);
    if (run.disabled) run.textContent = "执行中…";
    box.append(run);
  }
  return box;
}
async function importFile(id, kind, file) {
  if (!file) return;
  const token = epoch;
  let result;
  try {
    const url = new URL("import", apiBase);
    url.search = new URLSearchParams({ directory, kind });
    const form = new FormData();
    form.append("file", file);
    const response = await fetch(url, {
      method: "POST",
      headers: userHeaders,
      body: form,
    });
    result = await response.json();
    if (!response.ok) throw Error(result.error);
  } catch (error) {
    const path = prompt(
      `上传失败：${error.message}\n如果文件已在服务器 input 内，输入相对路径以拷贝；否则取消。`,
    );
    if (!path) throw error;
    result = await request("copy", { kind, path });
  }
  if (token === epoch) await select(id, { asset: result.asset });
}
async function execute(id) {
  if (jobs.has(id)) return;
  await flushDrafts();
  jobs.set(id, { id: null, epoch });
  redrawMaterial(id);
  try {
    await saving;
    const token = epoch,
      dir = directory;
    const compiled = await request("compile", {
      target: id,
      revision: documentState.revision,
    });
    jobs.set(id, { id: compiled.run_id, directory: dir, epoch: token });
    const response = await fetch(new URL("prompt", comfyBase), {
      method: "POST",
      headers: { ...userHeaders, "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: compiled.prompt,
        prompt_id: compiled.run_id,
        client_id: clientId,
        partial_execution_targets: [id],
      }),
    });
    const data = await response.json();
    if (!response.ok) throw Error(JSON.stringify(data.error));
    redrawMaterial(id);
    status("已提交执行");
  } catch (error) {
    jobs.delete(id);
    redrawMaterial(id);
    throw error;
  }
}
async function complete(promptId, ok) {
  for (const [id, job] of jobs)
    if (job.id === promptId) {
      jobs.delete(id);
      if (job.epoch !== epoch) continue;
      if (ok) {
        const result = await request("run", { run_id: job.id });
        const data = await request("project", {});
        for (const [key, selection] of Object.entries(data.selections))
          if (selection.revision >= current(key).revision)
            documentState.selections[key] = selection;
        if (drafts.has(id)) await flushText(id);
        status(
          result.selected
            ? "结果已保存并选中"
            : "结果已保存至历史，保留了执行期间的手动选择",
        );
      } else status("执行失败或取消，原素材保留", true);
      redrawMaterial(id);
    }
}
function connect() {
  const url = new URL("ws", comfyBase);
  url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  url.search = new URLSearchParams({ clientId });
  const ws = new WebSocket(url);
  ws.onmessage = (e) => {
    if (typeof e.data !== "string") return;
    const msg = JSON.parse(e.data);
    if (
      [
        "execution_success",
        "execution_error",
        "execution_interrupted",
      ].includes(msg.type)
    )
      complete(msg.data.prompt_id, msg.type === "execution_success").catch(
        report,
      );
  };
  ws.onclose = () =>
    setTimeout(() => {
      connect();
      for (const job of jobs.values())
        fetch(new URL(`history/${job.id}`, comfyBase), { headers: userHeaders })
          .then((r) => r.json())
          .then((h) => {
            if (h[job.id])
              return complete(
                job.id,
                h[job.id].status?.status_str === "success",
              );
          })
          .catch(report);
    }, 2000);
}
async function refreshTemplates() {
  const data = await request("templates", {});
  $("templates").replaceChildren(
    ...data.items.map((name) => new Option(name, name)),
  );
}
$("open").onclick = () => open().catch(report);
$("refresh-templates").onclick = () => refreshTemplates().catch(report);
$("add").onclick = async () => {
  try {
    await flushDrafts();
    await saving;
    const name = $("templates").value;
    if (!name) throw Error("先在 ComfyUI 保存一个 Canvas 卡片");
    await request("card/add", { name });
    await open();
  } catch (error) {
    report(error);
  }
};
$("fit").onclick = () => {
  const card = documentState.cards[0];
  if (card) view = { x: 40 - card.x, y: 40 - card.y, z: 1 };
  schedule();
};
let pan;
viewport.onpointerdown = (e) => {
  if (e.target !== viewport && e.target !== world) return;
  pan = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
  viewport.setPointerCapture(e.pointerId);
};
viewport.onpointermove = (e) => {
  if (!pan) return;
  view.x = pan.vx + e.clientX - pan.x;
  view.y = pan.vy + e.clientY - pan.y;
  schedule();
};
viewport.onpointerup = () => {
  pan = null;
};
viewport.addEventListener(
  "wheel",
  (e) => {
    if (e.target.closest("textarea,select,input")) return;
    e.preventDefault();
    const rect = viewport.getBoundingClientRect(),
      x = e.clientX - rect.left,
      y = e.clientY - rect.top,
      z = Math.max(0.15, Math.min(2, view.z * Math.exp(-e.deltaY * 0.001)));
    view.x = x - ((x - view.x) * z) / view.z;
    view.y = y - ((y - view.y) * z) / view.z;
    view.z = z;
    schedule();
  },
  { passive: false },
);
window.addEventListener("resize", schedule);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopPlayer();
});
window.addEventListener("beforeunload", stopPlayer);
const channel = new BroadcastChannel("turing-material-workspace");
channel.onmessage = async (e) => {
  if (e.data.directory === directory) {
    try {
      await open();
    } catch (error) {
      report(error);
    }
  }
};
$("directory").value =
  new URL(location.href).searchParams.get("directory") || "materials/project";
open().then(connect).catch(report);
refreshTemplates().catch(report);
