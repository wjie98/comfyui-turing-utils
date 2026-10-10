/** Native widgets and explicitly activated media previews for material cards. */
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
import { bindMaterialCombo } from "../lib/material_combo.js";
import { action, report, request } from "./api.js";

let playing = null;
export const stopActivePreview = () => playing?.releasePlayer();

export function button(node, name, fn) {
  return node.addWidget("button", name, null, action(fn), { serialize: false });
}
export function field(node, name, type, value, fn, options = {}) {
  const w =
    type === "STRING"
      ? ComfyWidgets.STRING(
          node,
          name,
          ["STRING", { default: value, multiline: options.multiline ?? true }],
          app,
        ).widget
      : node.addWidget(
          type === "BOOLEAN" ? "toggle" : type === "COMBO" ? "combo" : "number",
          name,
          value,
          () => {},
          options,
        );
  w.value = value;
  w.callback = action(fn);
  w.options = { ...w.options, serialize: false };
  return w;
}
export function url(directory, asset, thumbnail = false) {
  return api.apiURL(
    "/turing/workspace/asset?" +
      new URLSearchParams({
        directory,
        asset,
        ...(thumbnail ? { thumbnail: "1" } : {}),
      }),
  );
}
export function controls(node, m, { select, importMedia, run }) {
  if (m.kind === "text")
    field(node, m.id, "STRING", m.selection.text || "", (value) =>
      select(node, m, { text: value }),
    ).label = m.name;
  else {
    if (["image", "video"].includes(m.kind)) {
      const img = new Image();
      img.loading = "lazy";
      img.style.cssText = "width:100%;height:100%;max-height:180px;object-fit:contain";
      if (m.selection.asset)
        img.src = url(node.properties.directory, m.selection.asset, true);
      node.posters ||= new Map();
      node.posters.set(m.id, img);
      const preview = node.addDOMWidget(`poster:${m.id}`, "image", img, {
        serialize: false,
      });
      preview.computeSize = () => [280, 180];
    }
    const w = field(
      node,
      m.id,
      "COMBO",
      m.selection.asset || "",
      (value) => select(node, m, { asset: value }),
      { values: [m.selection.asset?.split("/").at(-1) || ""] },
    );
    w.label = m.name;
    const directory = node.properties.directory;
    bindMaterialCombo(node, w, {
      key: `${directory}:${m.kind}`,
      list: async () =>
        (
          await request("history", {
            directory,
            kind: m.kind,
            limit: 0,
          })
        ).items,
      selected: () => m.selection.asset,
      select: action((asset) => select(node, m, { asset })),
    }).catch(report);
    button(node, "choose file to upload", () => {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = m.kind + "/*";
      input.onchange = action(async () => {
        if (!input.files[0]) return;
        await importMedia(node, m, input.files[0]);
      });
      input.click();
    });
    button(node, `${m.kind} · 预览`, () => {
      if (!m.selection.asset) return;
      playing?.releasePlayer();
      node.releasePlayer();
      const media = document.createElement(m.kind === "image" ? "img" : m.kind);
      media.style.cssText =
        "width:100%;height:100%;max-height:240px;object-fit:contain";
      if (m.kind !== "image") {
        media.controls = true;
        media.preload = "none";
        media.onloadedmetadata = () => {
          media.currentTime = node.trims[m.id]?.start || 0;
        };
        media.ontimeupdate = () => {
          const trim = node.trims[m.id] || {};
          if (trim.end && media.currentTime >= trim.end) {
            media.pause();
            media.currentTime = trim.start || 0;
          }
        };
      }
      media.src = url(node.properties.directory, m.selection.asset, m.kind === "image");
      node.player = media;
      playing = node;
      node.playerWidget = node.addDOMWidget("preview", m.kind, media, {
        serialize: false,
      });
      node.playerWidget.computeSize = () => [280, m.kind === "audio" ? 54 : 180];
      node.setSize(node.computeSize());
      node.visibility = new IntersectionObserver((items) => {
        if (!items[0].isIntersecting) node.releasePlayer();
      });
      node.visibility.observe(media);
    });
  }
  if (["video", "audio"].includes(m.kind)) {
    node.trims ||= {};
    node.trims[m.id] = { start: 0, end: 0 };
    for (const [key, label] of [
      ["start", "起点（秒）"],
      ["end", "终点（秒，0 为末尾）"],
    ])
      field(
        node,
        `${m.name} · ${label}`,
        "FLOAT",
        0,
        (value) => {
          node.trims[m.id][key] = value;
        },
        { min: 0, step: 1, precision: 2 },
      );
  }
  if (m.executable) button(node, `${m.kind} · 执行到这里`, () => run(node, m));
}

export function releasePlayer(node) {
  node.visibility?.disconnect();
  node.visibility = null;
  if (node.player) {
    node.player.pause?.();
    node.player.onloadedmetadata = null;
    node.player.ontimeupdate = null;
    node.player.removeAttribute("src");
    node.player.load?.();
    node.player.remove();
    const index = node.widgets.indexOf(node.playerWidget);
    if (index >= 0) node.widgets.splice(index, 1);
    node.playerWidget?.onRemove?.();
    node.playerWidget = null;
    node.player = null;
  }
  if (playing === node) playing = null;
}

export function clearPosters(node) {
  for (const poster of node.posters?.values() || []) poster.removeAttribute("src");
  node.posters?.clear();
}
