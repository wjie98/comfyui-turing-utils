# Canvas card contract

## Ownership

- workspace/endpoints.py and web/lib/canvas_ports.js: dynamic ports, stable IDs and ordering.
- workspace/cards.py: validate endpoints and substitute inputs; native ComfyUI flattens subgraphs.
- workspace/workflow_files.py: independent instance files, revisions, namespacing and external connections.
- workspace/store.py: immutable material files, SQLite selections and runs.
- workspace/compiler.py: execution cut at saved boundaries; preserve model preparation caching.
- workspace/cache.py: version-sensitive task-end eviction of owned intermediates only.
- workspace/nodes.py: material boundaries and private fresh/read/write adapters.
- workspace/routes.py: storage and preview API, not another executor.
- workspace/ui: independent virtualized page; no LiteGraph / Nodes 2.0 dependency.
- web/material_workspace.js: native commands and official graph import/export.
- examples/h3_material_card.json: native editable workflow template, not a JavaScript graph builder or model implementation.

## Files and stable identities

canvas.json format 2 owns card positions and connections. Each instance owns cards/<id>/workflow.json.
Its extra.turing_card.version = 1 holds a derived API prompt, source hash and input overrides.
The interface is derived on read, not stored as a second competing definition.
Unknown versions fail explicitly. Library files live in the ComfyUI user's canvas_cards directory.
Import copies a template; editing an instance cannot edit the template or another instance.

Exactly one Canvas Inputs and Canvas Outputs must remain after official flattening.
Ports have id, zero-based slot, name, type, kind (value or position), optional default/options.
Slots are contiguous. IDs survive reorder; UI order is not identity.
Inputs marker links bind every material once. Outputs only accepts material outputs, with matching
type/slot. Every material must be exported. Position markers never become computation data.
Project material keys are <instance UUID>:<stub_id>, never native execution paths.
Do not change material type under a stable ID. Disconnect ports before deleting or retyping them.

## Execution and edits

External binding > card override > internal endpoint test input > endpoint default.
Overrides disconnect internal test inputs when opened in the editor.
Recipe changes do not invalidate saved materials or auto-run downstream cards.
A material's connected value is an explicit local action. Upstream saved boundaries become readers.
Use native queue/validation/execution; no custom action language or second inference engine.

File hashes protect save-back, project revisions protect edits, selection revisions protect manual
changes from delayed generation. Validate batches before writing. Each file replacement is atomic;
multi-file power-loss recovery is not transactional. Stale workflows stay openable but cannot run.

Do not globally clear caches, pin model tensors or retain an extra model cache.
Preserve normal preparation signatures/offload behavior. Cleanup must not mask execution errors.
Test Classic, LRU and RAM-pressure modes.

## UI rules

- Classic LiteGraph is required; do not rely only on Nodes 2.0 widgets.
- Use native commands, graphToPrompt, loadGraphData, queue and media APIs.
- Snapshot link membership before moving slots; new LiteGraph derives it from current indices.
- Reuse ordinary model/LoRA/Attention/Chat nodes rather than duplicate global settings.
- Drafts outlive virtualized DOM; merge asynchronous selections by revision.
- Bound previews, mount only nearby cards, titles only when zoomed out.
- No video/audio source before play; one player, no loop, release on unmount/page hide.
- Filename/prefix history; code cleanup never removes user material files.

## Verification

Use ops/test-dev.sh, tests/browser/native_parameter_layout.mjs and
tests/browser/material_workspace.mjs in development.
The browser harness uses Playwright/Chromium. MATERIAL_VIDEO_FIXTURE enables cold/play/Range and
1000-card virtualization checks. Keep synthetic artifacts inside the development instance.
Event dispatch tests handlers, not physical pointer accuracy on every client.
Real-model quality, target GPUs and remote low-bandwidth responsiveness require separate measurements.
