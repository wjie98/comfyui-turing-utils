# Node configuration and presets

## Current ordinary-node policy (supersedes earlier simplification notes)

Only Multimodal Prompt Chat's former Options fields are advanced. Its
`system_prompt` is ordinary. All other ordinary public nodes expose their
parameters directly, including inherited ConvRot CLIP device settings.
DynamicCombo branches remain mode-dependent, not advanced sections.

The frontend no longer sorts ordinary nodes by advanced status, except Chat.
Schema input order, persisted widget order, socket IDs, defaults and computation
are unchanged by this restoration. Fused SeC/upscaler nodes retain their loader
fields followed by their existing apply inputs; retired entry points are not
reintroduced. Legacy flat noise-grid migration remains in place.

Canvas is outside this change: its Attention schema explicitly requests the
previous advanced grouping. SeC/Crop stable connected controls remain a separate
compatibility concern, not an advanced-parameter feature.

Restoration checklist: simple loaders/VAE/upscaler, media/ROI/mask prompts,
padding/segment output/prefix noise, SeC, Bernini and ordinary Attention all use
their existing definition order without advanced filtering. Chat retains its
ordinary parameter order and appends the former Options in their existing order
(JPEG quality stays inside the JPEG branch). No positional-value migration is
needed because definitions were not reordered.

## Completed follow-up 1–5

1. Real frontend 1.53.6 browser validation covers Sol/SLA switching, prefix
   controls, JPEG/PNG, noise grids, advanced visibility, connected subgraph
   inputs, serialization and reopening. A plugin-scoped bridge supplies the
   advanced flag expected by the legacy canvas and excludes hidden advanced
   controls from its layout. Use the native node menu's advanced toggle.
   Saved sizes are preserved: resize-to-content can compact older tall nodes.
   No global settings cache, separate sidebar or model copy is added.
2. Three reusable subgraph workflows are provided below. No Python mega-node
   for references or Padding/Encode/Mask/AV was introduced.
3. Video Frames Padding shares the six video-mask types: Wan/Hunyuan/Hunyuan 1.5
   use 4*n+1; LTXV 8*n+1; Mochi 6*n+1 with minimum seven; H3 17*n+5.
   These are frame grids, not full VAE/conditioning compatibility claims.
4. Prefix Context Noise exposes block_size only in custom-grid mode. Legacy
   flat API arguments remain accepted. UI migration preserves custom-grid
   input slots/links, including inside subgraphs. Noise values are unchanged.
   Fixed-grid mode continues to ignore block size.
5. Audio protect_all and generate_all no longer require trim metadata.
   protect_prefix_generate_body still validates it. Video and audio temporal
   mappings remain separate.

## Reusable presets

Load the JSON workflow, then copy its subgraph into your workflow. External
model/material inputs are intentionally unconnected. These are expandable
building blocks, not standalone generation workflows with selected model files.

| File | Inputs / behavior | Outputs |
|---|---|---|
| [H3 shared references](../examples/h3_references_subgraph.json) | CLIP, prompt, target latent, optional encoded first/last/image/video/audio references; each reference feeds semantic encoding and conditioning assembly | conditioning |
| [H3 video preparation](../examples/h3_video_subgraph.json) | images, video VAE, optional image-frame masks; separate image-padding and mask-padding branches | latent, original length, padded length |
| [H3 AV preparation](../examples/h3_av_subgraph.json) | video preparation plus audio, audio VAE and mask policy; trim_info is last | AV latent, original length, padded length |

References are encoded outside the shared-reference preset. This avoids eagerly
loading VAEs for unused reference types and supports independent material reuse.
For ordinary fresh ref2va generation the core MiniMaxH3ReferenceToVideo already
assembles conditioning and an empty AV latent; this preset instead targets an
existing latent, useful for editing. Unpack it when semantic and DiT reference
selections must differ. No second-pass/upscaler is assumed.

Preparation masks are image-frame masks. A single mask repeats to padded length;
otherwise provide masks matching source frame count. Already-latent-time masks
should go directly to Set Video Latent Noise Mask outside the preset. Missing
masks preserve latent metadata; fresh VAE output has no mask and is unrestricted.
Spatial resizing is not part of preparation.

AV defaults to protect_all. Audio is required, not silently replaced by generated
audio. Align source duration explicitly: video padding does not invent matching
audio samples. Only prefix protection requires trim_info. Choose video-only
preparation to avoid unnecessary audio VAE execution.

Real PromptExecutor tests confirm mask edits do not re-encode video, strategy
edits do not reload the upstream model, and fixed chat inputs reuse cache while
advanced edits invalidate it. Browser checks confirm all presets reopen and
flatten to the expected internal API graph. These are graph/cache checks, not
new end-to-end model-quality benchmarks.

Verification note (2026-10-08, after entry cleanup): targeted tests passed
(116 tests, 106 subtests). The wider selected suite passed 688 tests and
1178 subtests after excluding CUDA/quantization selections and the VAE INT8 FP16
test file. Two existing strict numerical-equality checks in that file fail even
when run alone: 36-block decoder scoped rotation and FP32 fallback equality.
Their compute paths were not changed by this UI/preset work; tolerances were not
relaxed. Do not treat this result as a clean full numerical/kernel regression.

## Simplification boundaries

- Unified attention offers Sol/SLA/Veda under Models; individual strategy IDs,
  Image Sol, Static Virtual KV and experimental Block Cache are removed.
- Chat settings are inline advanced inputs; the old Options node/socket is removed.
- INT8 forcing, H3 decode backend, SeC backend and upscaler precision remain
  advanced, with unchanged defaults.
- Old model loader/apply IDs are removed. Five private, dev-only execution
  nodes preserve shared loading and independent cache boundaries.
- Public menus have one level beneath Turing Utils; see the inventory.
- Keep load/save/merge, crop/stitch, continuation/trim and prefix noise separate.
  Their different workflow positions and independent reuse are intentional.
- Ordinary workflows retain these composable boundaries. The separate
  [experimental Material Canvas](material-canvas.md) now provides task cards;
  it deliberately cannot mix with ordinary nodes.
