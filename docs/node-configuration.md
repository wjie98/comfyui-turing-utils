# Native node configuration contract

## Ordinary nodes

ComfyUI owns parameter order, widget/socket binding, layout, connection state,
serialization, search and menus. Backend schemas are the single source of
parameter definitions. Do not patch these to implement our own visual conventions.
DynamicCombo remains appropriate for genuinely mode-dependent input branches.

Configure Attention Strategy selects Sol, SLA, Veda, Spark or `none` with DynamicCombo.
Sol/SLA/Veda preserve an explicit upstream dense backend, including native
Kitchen; models without one default to W8A8. `none` preserves upstream execution
and adds no strategy. Spark requires W8A8 and rejects incompatible explicit
backends before sampling. The same resolved dense provider is used for H3 text
refinement, independently of the packed video/audio sparse strategy.
The optional top-level `vpack` combo is independent of strategy and selects the
reference-frame fraction to keep: `5/5` (default, disabled), `4/5`, `3/5`, or `2/5`.
Old off/drop-ratio workflows migrate to the matching keep ratio without changing
strategy or protection settings. Workflows predating VPack remain disabled.
Compression affects H3 Ref2VA reference videos before memory planning; protection
controls then operate on the retained sequence. Original first/last reference
anchors remain available to `sparse_reference_image`. Manual prefix counts refer
to the compressed sequence. Target video/mask, reference audio and Qwen results
are unchanged. `strategy=none` can still use VPack while inheriting attention.
Spark and Veda require kernel 0.48.0 and expose the same protection controls
as Sol/SLA: `prefix_policy=auto|none|manual`. Auto shows
`sparse_reference_image`, `sparse_reference_video`, and `sparse_reference_audio`;
manual shows `manual_prefix_tokens`, measured in the packed sequence after VPack.
False protects both Query and K/V; true permits sparse routing. Image includes
keyframes and the first/last frames of reference video. Audio affects reference
audio only. Text, target audio and unsupported layouts remain dense.

Spark defaults to auto with all three reference switches false,
`topk_ratio=0.20`, `dense_prefix_steps=2`, and the other dense step/layer counts
zero. It uses W8A8 independently of the loader. Complete Landmark-v2 clustering
is applied separately to unprotected spans; protected tokens never enter the
clustering permutation. Mixed 64-token boundary blocks stay exact. Top-K counts
only unprotected candidate blocks; protected K/V is additional.

Veda defaults to auto with image/audio false and video true. Keep ratios of zero
use the predictor's trained ratios, and all dense step/layer counts default to
zero. Protection preserves the trained 3D tile geometry: touching a protected
token protects the complete 128-token tile in both directions. Protected tiles
are excluded from candidate budgets. `reference_keep_ratio=1` overrides sparse
permissions and keeps every reference dense in both directions. Reference audio
can use sequential 128-token tiles when enabled; this optional route is not a
claim that the visual predictor was trained for audio. Keep it disabled unless
the speed/quality tradeoff is validated for the intended audio task.

None removes extra semantic protection; manual replaces auto with `[0,N)`.
Both retain each algorithm's dense-only modalities. Dense steps follow Sol's
whole-task H3 timeline, without ratios or a restart at upscale. The old Spark
backend, warmup and debug fields remain removed. Recreate affected Spark/Veda
nodes saved before the protection controls were added; no migration shim is
installed.

SeC positive_coords and negative_coords are optional socket-only STRING inputs,
in that order immediately before bounding_box. Use native force_input rather
than JavaScript widget hiding; their JSON point format is unchanged.

Only Multimodal Prompt Chat's former Options parameters are advanced;
system_prompt and existing ordinary-node inputs remain ordinary.
H3 KSampler keeps model/conditioning/final-size latent, seed/sampler and the
main curve before its native DynamicCombo mode selector. Main curves use
scheduler/manual only. Default simple steps=4 plus tail_steps=1 replaces 4+1;
0 tail steps preserves native scheduling exactly. Tail points subdivide the
last interval in the unshifted clock and follow the current model's shift.
Modes are single (no additional controls), progressive (lowres scale/steps,
upscaler and high mask growth), and two_pass (lowres scale, second curve,
upscaler and high mask growth). Second curves use shift-aware LBH examples,
scheduler, or absolute manual values. Upscaler precision is auto. The worker's
lazy loader socket stays top-level because ComfyUI's dependency planner does
not discover lazy flags nested inside DynamicCombo inputs; this internal socket
is not part of the public UI. H3 has no advanced, negative, CFG, second latent,
second condition or high-resolution MODEL inputs. Old workflows need the explicit
final-canvas migration described in README; no global widget hooks are added.
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

The serialized schema baseline covers ordered input names, DynamicCombo branches,
defaults and output types. Native queue tests exercise workflow execution and
shared loaders. The legacy VPack drop ratio is translated at prompt submission.
