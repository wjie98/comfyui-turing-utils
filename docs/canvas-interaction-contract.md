# Canvas interaction contract

Canvas is a material editor over existing Turing Utils / ComfyUI nodes, not a
second inference engine. Current schema version is 3.

## Ownership and controls

| Card | Controls and purpose | Existing implementation to reuse |
| --- | --- | --- |
| Root | work/cache location, default 4 MP material limit; save/open project | ComfyUI instance directories and native string editing |
| H3 Settings | model files, LoRAs, dense backend, optional attention strategy, sampling and Chat configuration | ConvRot DiT/CLIP, Configure Attention Strategy, sigma shift, multimodal Chat |
| Image | project filename, import, per-material MP limit (0 inherits Root) | immutable original plus bounded decode |
| Video | project filename, time in/out, MP limit, include audio | 24 FPS timeline with synchronous audio cropping |
| Audio | project filename, time in/out | same selection timeline as Video |
| H3 Generate | user/model prompts, aspect/MP/width/height, duration, denoise, output prefix, target-audio preservation | Resolution Selector, frame padding, H3 references, VAE, latent masks, AV concat, scheduler/sampler |

Global queue never executes Canvas. Generate explicitly queues only the selected
card. Generation-parameter edits do not change a published output; successful generation
selects a new file, failure leaves the old result selected. Restore parameters
never queues work, rewires links, changes project location or switches output.
H3 restores its last successful generation parameters; settings/material cards
restore their last explicitly saved project parameters.

## Material and modality contract

Image, video and audio have distinct Canvas-only socket types and colors. A video
file is a container: video references consume pictures only; audio references
consume soundtrack only. Prefix/target consume both unless include_audio is off.
Missing/disabled audio is an actionable error on an audio-only connection, not
invented silence. Prefix remains temporary context and is excluded from output.
Original files are immutable. Time range and decode limits travel with references.
Generate has the same output selection as imported video: adjusting its trim range
changes downstream material signatures without re-running inference or rewriting
the complete historical file. A single video output connects to video or audio
inputs via ComfyUI MultiType; it carries both modalities only for prefix/target.
Display numbering starts at 1; existing zero-based port IDs remain stable.

## Sampling, LoRA and import

H3 Settings exposes sampler, scheduler, base steps and a default-on Refiner.
BasicScheduler handles partial denoise before refinement. The refiner redistributes
the existing tail beginning at the first sigma <=0.7 using cosine interpolation,
adding one sample point. No nonterminal sigma <=0.7 means no refinement; denoise=0
skips all sampling. No manual sigma field or presets. This follows the default
[H3SigmaRefiner algorithm](https://github.com/yichengup/ComfyUI-YCNodes-MiniMax-H3/blob/main/py/h3_sigma_refiner.py),
not a second pass or upscaler; visual benefit still requires model validation.

LoRA rows have enabled/name/strength/remove controls and retain their order.
Their JSON is storage-only; disabled and zero-strength rows are omitted from the
private execution graph. Labels use owner namespaces (ConvRot Loader, Attention
Strategy, H3 Sigma Shift, Sampling, Prompt Chat) without renaming persisted keys.

Browser imports try streamed upload first, then server input-directory copy using
explicit local_path or the browser basename, with a size check. This cannot recover
a browser computer's path remotely; both failures are reported. Entering an
explicit local_path and pressing import directly copies that server file.

## Interaction and performance

Use native widgets and callbacks first, small DOM controls only for media/time
selection. Never hijack string-field clicks. Directory browsing reuses one dialog
and lists one level asynchronously. Empty-tree deletion rejects all files,
symlinks and special entries; use rmdir only, never recursive file deletion.
No idle polling or full-media downloads for previews. Range controls update local
state while dragging; commit one refresh on release. Directory/probe work stays
off the server event loop. Do not add independent model-weight caches.
The media decoder stays native; a small custom timeline provides in/out handles,
playhead and selection-only playback. Previews grow with node size; no idle animation
loop, base64 video or full-media decode is introduced. Original playback still
costs source bandwidth. H3 and imported assets share this implementation.

Resolution presets use the official 1024²-pixels-per-MP convention and 32-pixel
alignment. Explicit width/height are canonical; manual edits update ratio and MP.
Duration is seconds at 24 FPS, with model frame padding internal and trimmed away.

## Evolution

### Reusable frontend controls

`web/canvas/ui.js` owns native choice menus, request lifetime guards and bounded
preview sizing. Use these helpers for new cards rather than per-node HTML
selectors or resize-delta accumulation. The preview layout minimum is constant;
node resize and restored dimensions are clamped, with a maximum preview height
of 900 pixels and node width of 1600 pixels.

LoRA rows are Canvas widgets beside ordinary controls, not floating DOM overlays.
Only the JSON storage widget is persisted; visual rows are rebuilt from it.
Material/history selectors use native combo widgets and project-relative filenames.
History listing must not create project directories. Selection errors retain the
previous successful output. Pending selections prevent generation until settled.

Async UI work must validate node identity, project and request revision before
applying results. Clearing a material releases its preview source and timeline.
History lists invalidate on import/generation, not idle polling. Preserve stable
serialized widget/socket order when changing display order (see `widget_layout.js`).

Ordinary Turing nodes display common controls before advanced controls without
reordering their serialized widgets. Dynamic children synchronize advanced flags
at layout time, not just node creation. Internal execution nodes remain registered
for API execution, but `turing.internal` filters the search/library and a permanent
`skip_list` hides the native context-menu entry even with ComfyUI developer mode
enabled. Internal schema titles and display mappings must explicitly say Internal;
public fused nodes must override any inherited internal title.

`tests/browser/canvas_controls.mjs` checks native control contracts, LoRA persistence,
layout and repeated preview resize against a running development server. It requires
external Playwright/Chromium (PLAYWRIGHT_MODULE and CHROMIUM_PATH); do not install
browser dependencies into production. Widget-handler tests are not a replacement
for manual dragging/clicking in an active ComfyUI workflow tab.

Persist schema version and named widget values, never rely on positional widget
arrays for new saves. Maintain explicit migration for supported older schemas;
Version 3 removes import_mode/manual sigmas and folds old audio-output links into
video output 0, preserving destination modality. Removed sigma values are not used.
reject future versions instead of silently reinterpreting data. Legacy frames,
Sol and time-selection values are normalized at the Canvas boundary. Existing
v1 video sampling/resize options are carried in hidden compatibility metadata;
editing a time range replaces its old frame-sampling options, editing the MP
limit replaces old custom dimensions. No new controls expose those old options.
Do not
rename persisted socket IDs just to change labels. New cards must document their
material types, modality selection, parameter ownership, restore checkpoint,
existing-node reuse, and version migration, with contract and UI tests.

Do not delete or silently rewrite old material files. UUID-era projects preceding
the filename layout still require explicit import, as documented previously.
