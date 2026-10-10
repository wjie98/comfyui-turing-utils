/** Workspace transport and native ComfyUI feedback. */
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

export async function request(path, body = {}) {
  const r = await api.fetchApi(`/turing/workspace/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) throw Error(data.error || JSON.stringify(data));
  return data;
}
export const report = (e) =>
  app.extensionManager.toast.add({
    severity: "error",
    summary: "Canvas",
    detail: e.message || String(e),
    life: 7000,
  });
export const action =
  (fn) =>
  (...args) =>
    Promise.resolve()
      .then(() => fn(...args))
      .catch(report);
export const prompt = (title, value = "") =>
  app.extensionManager.dialog.prompt({
    title,
    message: title,
    defaultValue: value,
  });
