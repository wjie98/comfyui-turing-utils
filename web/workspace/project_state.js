/** Per-project save ordering and lightweight native graph snapshots. */
import { app } from "../../../scripts/app.js";
import { request } from "./api.js";

export const ROOT = "TuringCanvasProject",
  CARD = "TuringCanvasCard",
  jobs = new Map();
const saves = new Map();
export const project = () => app.graph.extra?.turing_project;
export const editor = () => app.graph.extra?.turing_editor;
function projectState(directory) {
  if (project()?.directory === directory) return project();
  return app.extensionManager.workflow.openWorkflows.find(
    (workflow) => workflow.activeState?.extra?.turing_project?.directory === directory,
  )?.activeState?.extra?.turing_project;
}
function snapshot(directory) {
  if (project()?.directory === directory) return app.graph.serialize();
  const tab = app.extensionManager.workflow.openWorkflows.find(
    (w) => w.activeState?.extra?.turing_project?.directory === directory,
  );
  return tab?.activeState
    ? { ...tab.activeState, nodes: [...tab.activeState.nodes] }
    : null;
}
export function enqueueSave(state, operation) {
  // Callers may capture a state before awaiting another save (card insertion
  // does this). The live tab owns the latest revision, not an idle queue cache.
  const revision = Math.max(
    state.revision,
    projectState(state.directory)?.revision ?? state.revision,
  );
  let queue = saves.get(state.directory);
  if (!queue)
    saves.set(
      state.directory,
      (queue = { revision, tail: Promise.resolve(), pending: 0 }),
    );
  queue.revision = Math.max(queue.revision, revision);
  queue.pending++;
  const task = queue.tail.then(async () => {
    const result = await operation(queue.revision);
    queue.revision = result.revision;
    state.revision = result.revision;
    if (project()?.directory === state.directory) project().revision = result.revision;
    for (const tab of app.extensionManager.workflow.openWorkflows) {
      const p = tab.activeState?.extra?.turing_project;
      if (p?.directory === state.directory) p.revision = result.revision;
    }
    return result;
  });
  queue.tail = task.catch(() => {});
  return task.finally(() => {
    queue.pending--;
    if (queue.pending === 0 && saves.get(state.directory) === queue)
      saves.delete(state.directory);
  });
}
export async function saveProject(directory = project()?.directory) {
  if (typeof directory !== "string") directory = project()?.directory;
  if (!directory) throw Error("当前不是画布");
  const state = projectState(directory);
  if (!state) return; // A closed project already saved before task submission.
  setProjectStatus(directory, "正在保存");
  try {
    const result = await enqueueSave(state, (revision) => {
      // Capture at queue execution, not enqueue time: adding a card may have
      // changed the native graph while this autosave was waiting.
      const current = snapshot(directory);
      if (!current) return { revision };
      current.nodes = current.nodes.map((n) => ({
        id: n.id,
        type: n.type,
        pos: n.pos,
        size: n.size,
        properties: { instance: n.properties.instance },
      }));
      return request("project/save", {
        directory,
        revision,
        workflow: current,
      });
    });
    setProjectStatus(directory, "已保存", result.statistics);
  } catch (error) {
    setProjectStatus(directory, "保存失败");
    throw error;
  }
}
export function setProjectStatus(directory, status, statistics, root) {
  root ||=
    project()?.directory === directory && app.graph._nodes.find((n) => n.type === ROOT);
  if (!root) return;
  if (statistics) root.properties.statistics = statistics;
  const stats = root.properties.statistics;
  const info = root.widgets.find((w) => w.name === "画布统计");
  if (info && stats) {
    const counts = Object.entries(stats.materials)
      .map(([kind, count]) => `${kind}: ${count}`)
      .join(" · ");
    info.value = `${stats.cards} 张卡片 · ${counts} · ${(stats.bytes / 1048576).toFixed(1)} MiB`;
  }
  const saved = root.widgets.find((w) => w.name === "保存状态");
  if (saved)
    saved.value = `${status} · ${[...jobs.values()].filter((j) => j.directory === directory).length} 个任务`;
  root.graph?.setDirtyCanvas(true, true);
}
