# Material Canvas (experimental)

Six cards live under `Turing Utils / Canvas`. Ports carry project-relative material
filenames, not ordinary IMAGE/AUDIO tensors. Mixed workflows are rejected.

## Cards

- **Canvas Root**: one per project; work/cache directories, browser upload or
  server input-directory copy, maximum decoded material megapixels (default 4). The directory
  picker browses this server instance's output, not the browser computer.
- **Canvas H3 Settings**: global models, editable LoRA stack, sampler/scheduler/base steps,
  default-on low-noise Refiner, shifts, attention,
  and Chat configuration (system prompt, URL, model and API-key environment name).
  Attention strategy directly reuses Configure Attention Strategy: disabled,
  Sol, SLA or Veda, with the same dynamic advanced controls and model requirements.
- **Canvas Image / Video / Audio**: import or drag files into the project; select
  existing project material by filename without copying again. Video supports
  a 24 FPS decode timeline, start/end seconds and draggable time-range controls;
  audio has the same time selection. End 0 means to the end. Image/video MP 0
  inherits Root. Video include_audio defaults on; switching off suppresses audio.
- **Canvas H3 Generate**: first/last, dynamic image/video/audio references,
  optional prefix and target videos. Duration replaces frame count; aspect ratio
  and megapixels update 32-aligned width/height, and manual dimensions update
  ratio/MP. MP follows Resolution Selector's 1024² convention.
  No mask, mode, seed or upscale controls.
  The button between prompts explicitly generates the model prompt. Generation
  uses nonblank model prompt, otherwise user prompt. Generate is at the bottom.
  Each run uses a new internal seed recorded in result metadata.

## Target, prefix and sampling

`denoise` uses native BasicScheduler/KSampler semantics, not sigma scaling.
1 fully redraws; partial denoise takes the tail of a longer schedule; 0 skips
sampling and DiT/CLIP execution. Target still passes through VAE encoding/decoding,
so zero is not pixel-lossless file copying. Refiner adds one step by cosine
redistribution of the existing <=0.7 sigma tail, if such a nonterminal tail exists.
Base 4/8 steps normally become 5/9; there are no manual sigma presets.

Target supplies the encoded body and its selected frame count. Without target,
the body uses empty latent content and the requested count. Prefix is additional
protected context, including its soundtrack when present. Existing video padding,
mask mapping and audio-prefix protection are reused. Results exclude prefix and
repeated padding frames. Target audio is preserved by default; absent body audio
is generated.

Video selection preserves the source time axis instead of speeding up audio.
Pictures are normalized to H3's 24 FPS timeline at decoding. Maximum material pixels are applied
during reads without changing stored originals. H3 reference encoders may resize
further. The integrated preview timeline plays only the selected time range.
Generate also exposes output trim: downstream decoding uses the selected range,
while the complete generated historical file stays unchanged.

## Files and history

```text
work/
  canvas.json
  parameters.json
  project.json
  cache.json
  materials/
    images/portrait.png
    images/portrait.png.json
    videos/source.mp4
    audio/voice.wav
  generations/
    shot_a/shot_a_000001.mp4
    shot_a/shot_a_000001.mp4.json
```

Import preserves filenames; collisions append a number without overwriting.
filename_prefix chooses the generation folder and basename. History lists only
completed results with that prefix. Sidecars record inputs, prompts, effective
seed and model/sampler settings, excluding Chat configuration and secrets.
References are relative, so the whole project can move within its owning output.

Work directories stay under this instance's output, caches under .cache,
local-copy sources under input. No sharing across instances or automatic cleanup.
This layout replaces the experimental UUID layout: old files are not deleted or
silently migrated. Start a new project and explicitly import old results.

## Execution and UI

Material changes mark downstream results pending. Changing models/prompts/settings
does not invalidate existing results. Explicit Generate always starts a new run.
Success selects the new output; failure retains the previous result.

Global execution is disabled: use each card's Generate button. Restore parameters
on H3 loads its last successful run; other cards restore the last explicitly saved
project checkpoint. Neither operation changes links or executes inference.
Right-corner badges show pending, missing,
running or failed; ordinary material cards have no badge.

Idle fixed polling is removed. Changes debounce state refresh; running tasks
still check completion. State responses omit historical snapshots. Images use
cached 512px JPEG thumbnails. Video first loads its thumbnail and fetches the full
stream only on playback. Full playback still uses source-video bandwidth.
Uploads stream to disk; proxy limits still apply and uploads are not resumable.
Failed uploads attempt a server input-directory copy (local_path, or basename),
checking size. Remote browser files cannot be copied without a server-side source.

Image/video/audio socket colors differ. Display numbering starts at 1 while
internal zero-based port IDs remain stable. Video-reference connections carry
pictures only; connect the same video output to an audio input for soundtrack conditioning.
Prefix/target consume both enabled modalities. Audio-only connections reject
missing or disabled soundtracks.

The work-directory text field remains editable; its adjacent folder button opens
one reusable asynchronous browser. Empty-directory deletion checks the complete
tree and refuses files/links; it never deletes material or project metadata.
Browsing/state refresh no longer creates project directories.

Named parameter persistence and schema versions are described in
[the interaction contract](canvas-interaction-contract.md). That document is the
checklist for future Canvas cards and compatibility changes.

Model-backed generation still requires manual validation with the intended weights.
Tests cover storage, compilation, CPU media round trips and mocked VAE boundaries;
they do not establish generation quality or large-canvas frame-rate guarantees.
