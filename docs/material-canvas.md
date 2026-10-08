# Material Canvas (experimental)

Six cards live under `Turing Utils / Canvas`. Ports carry project-relative material
filenames, not ordinary IMAGE/AUDIO tensors. Mixed workflows are rejected.

## Cards

- **Canvas Root**: one per project; work/cache directories, browser upload or
  server input-directory copy, maximum decoded material megapixels. The directory
  picker browses this server instance's output, not the browser computer.
- **Canvas H3 Settings**: global models, LoRAs, sampler steps, shifts, attention,
  and Chat configuration. Sol is disabled by default; enabling it reveals routing,
  dense steps/layers and reference sparsity. API secrets use environment names.
- **Canvas Image / Video / Audio**: import or drag files into the project; select
  existing project material by filename without copying again. Video supports
  dimensions, rate, skipped frames, frame cap and sampling interval; audio has
  start/duration controls. Zero duration means to the end.
- **Canvas H3 Generate**: first/last, dynamic image/video/audio references,
  optional prefix and target videos. No mask, mode, seed or upscale controls.
  The button between prompts explicitly generates the model prompt. Generation
  uses nonblank model prompt, otherwise user prompt. Generate is at the bottom.
  Each run uses a new internal seed recorded in result metadata.

## Target, prefix and sampling

`denoise` uses native BasicScheduler/KSampler semantics, not sigma scaling.
1 fully redraws; partial denoise takes the tail of a longer schedule; 0 skips
sampling and DiT/CLIP execution. Target still passes through VAE encoding/decoding,
so zero is not pixel-lossless file copying. Clear custom sigmas before using
partial denoise; an explicit trajectory cannot be extended by a scheduler.

Target supplies the encoded body and its selected frame count. Without target,
the body uses empty latent content and the requested count. Prefix is additional
protected context, including its soundtrack when present. Existing video padding,
mask mapping and audio-prefix protection are reused. Results exclude prefix and
repeated padding frames. Target audio is preserved by default; absent body audio
is generated.

Video selection preserves the source time axis instead of speeding up audio.
Pictures are normalized to H3's 24 FPS timeline at decoding; force_rate is a
source-selection control, not output FPS. Maximum material pixels are applied
during reads without changing stored originals. H3 reference encoders may resize
further. Preview shows the original material, not a rendered edit of its selection.

## Files and history

```text
work/
  canvas.json
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

Global execution is initially locked; allow it once, inspect changed nodes, then
confirm. It locks before execution. Right-corner badges show pending, missing,
running or failed; ordinary material cards have no badge.

Idle fixed polling is removed. Changes debounce state refresh; running tasks
still check completion. State responses omit historical snapshots. Images use
cached 512px JPEG thumbnails. Video first loads its thumbnail and fetches the full
stream only on playback. Full playback still uses source-video bandwidth.
Uploads stream to disk; proxy limits still apply and uploads are not resumable.

Model-backed generation still requires manual validation with the intended weights.
Tests cover storage, compilation, CPU media round trips and mocked VAE boundaries;
they do not establish generation quality or large-canvas frame-rate guarantees.
