# Native node configuration contract

## Ordinary nodes

ComfyUI owns parameter order, widget/socket binding, layout, connection state,
serialization, search and menus. Backend schemas are the single source of
parameter definitions. Do not patch these to implement our own visual conventions.
DynamicCombo remains appropriate for genuinely mode-dependent input branches.

SeC positive_coords and negative_coords are optional socket-only STRING inputs,
in that order immediately before bounding_box. Use native force_input rather
than JavaScript widget hiding; their JSON point format is unchanged.

Only Multimodal Prompt Chat's former Options parameters are advanced;
system_prompt remains ordinary. web/chat_advanced.js opts Chat into the small
enableAdvancedLayout helper in web/lib/advanced_layout.js. On classic frontend
1.53.6, construction sets options.advanced but drawing reads widget.advanced;
native layout also includes advanced widgets even while drawing omits them.
The helper synchronizes flags and excludes folded advanced widgets from layout.
It never mutates widget.hidden, input slots, values, order or node size. The
native toggle, sizing, serialization and rendering remain in charge. Installation
is idempotent and per-node, not a global prototype patch. Remove this bridge when
the supported native frontend handles both flags and folded layout consistently.

There is no old-workflow migration or detached-socket repair layer. Workflows
saved with previous custom layouts may need affected nodes recreated. Do not
reintroduce broad compatibility hooks to preserve those broken layouts.

H3 Keyframe Reference has three fixed optional image inputs and three matching
outputs. Empty positions remain empty. Stage Barrier has eight fixed optional
value inputs and eight matching outputs. Its execution scheduling/compiler is
retained; frontend socket growth and output synchronization are removed.

## Internal nodes

Every implementation-only node must use native is_dev_only (V3) or DEV_ONLY
(V1), a display name ending in (Internal), and the shared INTERNAL_NODE_NOTE
in its description. Developer mode may intentionally expose these nodes.
No custom search interception, palette filtering, or strict hiding is needed.
Public fused nodes must not inherit the internal description marker.

## Material Workspace

The independent workspace retains its material persistence and execution UI,
explicit endpoint sorting, previews and workflow editing commands. These are
user-facing capabilities, not ordinary-node layout replacements.

Material types are declared once in workspace/protocol.py. API graph interfaces
are derived rather than separately persisted. H3's starting workflow lives in
examples/h3_material_card.json and is loaded with app.loadGraphData; do not
recreate its graph with a parallel JavaScript builder. Hidden endpoint metadata
uses native hidden/socketless input declarations.

Use small shared functions for real repeated behavior. A one-off fix does not
justify a widget framework or overrides on every node prototype.

## Reusable ordinary workflows

- examples/h3_references_subgraph.json: shared semantic/conditioning references.
- examples/h3_video_subgraph.json: frame padding and video VAE encoding.
- examples/h3_av_subgraph.json: video preparation plus audio/mask policy.

These remain editable workflows, not new Python mega-nodes. Image-frame masks
must match source length, or contain one mask for repetition. Latent-time masks
belong directly on the latent. AV protect_all requires audio; only prefix
protection requires trim metadata. Prefix Context Noise's custom block size is
inside its native DynamicCombo branch; old flat arguments are not migrated.

## Verification

tests/browser/native_parameter_layout.mjs compares each ordinary public node
against native registration of the same schema: order, widget/socket geometry,
DynamicCombo modes, connections, resizing and serialized reloads. Crop and SeC
also have dedicated link and disabled-editor tests. Chat checks the native
advanced toggle. Run these on the classic canvas, not only Nodes 2.0.

tests/browser/chat_advanced_layout.mjs additionally compares actual widget
positions, allocated heights and free resize space with native hidden-widget
layout across sizes, native toggles, JPEG/PNG branches and serialized reloads.
Testing draw visibility alone does not catch invisible widgets reserving space.

Material browser tests cover workflow templates, endpoint edits, nested
subgraphs, persistence and bounded playback. Tests must check actual links and
geometry, not just whether controls remain visible. Passing the tested frontend
is not a claim that every historical frontend or third-party extension works.
