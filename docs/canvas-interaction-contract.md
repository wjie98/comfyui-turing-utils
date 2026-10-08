# Canvas interaction contract

Canvas is a material editor over existing Turing Utils / ComfyUI nodes, not a
second inference engine. Changes below establish schema version 2.

## Ownership and controls

| Card | Controls and purpose | Existing implementation to reuse |
| --- | --- | --- |
| Root | work/cache location, upload/local copy, default 4 MP material limit; save/open project | ComfyUI instance directories and native string editing |
| H3 Settings | model files, LoRAs, dense backend, optional attention strategy, sampling and Chat configuration | ConvRot DiT/CLIP, Configure Attention Strategy, sigma shift, multimodal Chat |
| Image | project filename, import, per-material MP limit (0 inherits Root) | immutable original plus bounded decode |
| Video | project filename, time in/out, MP limit, include audio | 24 FPS timeline with synchronous audio cropping |
| Audio | project filename, time in/out | same selection timeline as Video |
| H3 Generate | user/model prompts, aspect/MP/width/height, duration, denoise, output prefix, target-audio preservation | Resolution Selector, frame padding, H3 references, VAE, latent masks, AV concat, scheduler/sampler |

Global queue never executes Canvas. Generate explicitly queues only the selected
card. Parameter edits do not change a published output; successful generation
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
Display numbering starts at 1; existing zero-based port IDs remain stable.

## Interaction and performance

Use native widgets and callbacks first, small DOM controls only for media/time
selection. Never hijack string-field clicks. Directory browsing reuses one dialog
and lists one level asynchronously. Empty-tree deletion rejects all files,
symlinks and special entries; use rmdir only, never recursive file deletion.
No idle polling or full-media downloads for previews. Range controls update local
state while dragging; commit one refresh on release. Directory/probe work stays
off the server event loop. Do not add independent model-weight caches.

Resolution presets use the official 1024²-pixels-per-MP convention and 32-pixel
alignment. Explicit width/height are canonical; manual edits update ratio and MP.
Duration is seconds at 24 FPS, with model frame padding internal and trimmed away.

## Evolution

Persist schema version and named widget values, never rely on positional widget
arrays for new saves. Maintain explicit migration for supported older schemas;
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
