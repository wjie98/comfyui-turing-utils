# Qwen Image 2.1 optimization

This adapter only matches `QwenImage21Transformer2DModel` and its Qwen3-VL 8B
text encoder. It does not change H3/Wan, model weights, sampling steps, reference
resolution, image order, or token selection.

## Integration

`Load ConvRot DiT` installs the conservative model adapter automatically for Qwen
Image 2.1. `Load ConvRot CLIP` with type `qwen_image` enables ComfyUI's quantized
matmul context during Qwen 2.1 encoding, as well as during generation. The scope
is restored after encoding and does not apply to other text encoder classes.
This encoding path uses activation quantization; it is not numerically identical
to a dequantized, full-precision encoder. The matched DiT comparisons below use
the same quantized encoder conditions in every mode.

`Qwen Image 2.1 多图条件` (`TuringUtilsQwenImage21Conditioning`) accepts the clip,
prompt, optional VAE, resolution (default 1024) and up to 16 reference images.
It preserves the native resize, alpha handling, reference order and first-image
latent geometry, but encodes only the positive prompt. Use it with `BasicGuider`
or CFG 1. For a normal KSampler at CFG 1, connect its positive output to both
conditioning inputs. Use the native conditioning node for CFG greater than 1.
Resolution 0 preserves each reference's size rounded to multiples of 32.

`Qwen Image 2.1 多图优化` (`TuringUtilsQwenImage21Optimize`) can also attach the
model adapter after a native loader. Place it after LoRAs and before the guider
or sampler. It exposes:

| Input | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Enable the model adapter and QKV input sharing. Disable for a matched DiT baseline. |

Keep the native `Qwen Image 2.1 Cache` node for cache device and storage dtype
selection. It is an upstream experimental node. The plugin's two Qwen Image
2.1 nodes are regular nodes. Setting the native cache device to `off` disables
prefix reuse.

Compatible BF16 ConvRot W8 Q/K/V projections share one activation quantization.
Projection hooks, bypass LoRA forwards, input smoothing scales, or full-precision
matmul requests use the native projection path. ComfyUI versions with writable
DynamicVRAM KV slots retain their native model forward and cache lifecycle.
The older ComfyUI path also skips unused reference and text projections and
position-table construction after a prefix cache hit. ComfyUI owns cleanup on
both paths.

## Local verification

Run tests through `ops/test-dev.sh`. Coverage includes multiple references,
cache invalidation, slot reuse, non-square targets, hooks, bypass-LoRA fallback
and encoder scope. Positive-only conditioning is compared
with the native node for multiple references, RGBA, non-square resizing, VAE
references and vision-only inputs.

On ComfyUI 0.46.0, a real two-reference 512-square generation on an A40 at
CFG 1 and four steps verified that both modes hit the native DynamicVRAM KV
cache. The optimized path shared QKV input quantization and produced the same
latent and PNG as the native path, with allocator graph compilation disabled
in both modes.

After removing the computation-cache prototype: 1396 passed, 143 skipped,
1457 subtests passed.

The installed kernel and checkout both report 0.48.0, with the installed package
pointing to the development kernel's editable project path. The retained changes
are plugin Python code and reuse existing kernel operations. No kernel rebuild,
CUDA extension ABI change or framework migration is needed.

`scripts/benchmark_qwen_image21.py` measures 25 real-weight DiT forwards with
synthetic conditions, including one prefix fill and 24 cached steps. It warms
each shape and disables the allocator graph equally for all modes. It is not
an image-quality or full-pipeline benchmark. Its default text length is 85.
`scripts/validate_qwen_image21.py`
runs actual multi-reference conditioning, Euler/simple sampling at CFG 1 and
VAE decoding, writing images and JSON records into the requested dev directory.
Run these from the development root using the `didv` environment and the same
CUDA/NVRTC preload and cache environment as `ops/run-dev.sh`. Pass
`--positive-only` to use the new conditioning node. Both scripts compare the
native and optimized paths; neither script downloads weights.

Final A40 results at 1024 square, 25 forwards, 85 text tokens, BF16 native KV
storage for the baseline and conservative modes:

| References | Native DiT | Conservative | KV cache |
| --- | ---: | ---: | ---: |
| 1 | 14.39 s | 14.17 s | 2.04 GiB |
| 2 | 17.44 s | 17.07 s | 4.04 GiB |
| 4 | 23.71 s | 23.37 s | 8.04 GiB |

The conservative final forwards are bit-identical in all three cases. These
measurements are recorded in `forward-benchmark-final.json`.

Real two-reference 1024-square generation, Euler/simple, CFG 1, 25 steps, seed
20261020, after two warmup sampling steps per mode:

| Mode | Sampling | KV storage | Output |
| --- | ---: | ---: | --- |
| Native | 17.44 s | 4.04 GiB | Baseline |
| Conservative | 17.28 s | 4.04 GiB | Bit-identical latent and PNG |

These are single matched runs, not statistical speedup guarantees. Sampling
includes prefix fill and excludes encoding, VAE decoding and cold model loading.
KV figures are cache bytes, not total GPU memory. Positive-only conditioning
took 10.90 s including initial reference/VAE work; this is not a controlled
encoder speed comparison. The first VAE decode took 0.89 s; subsequent decodes
took about 0.42 s.

Conservative output SHA-256:
`84f0772d352c66924f009c1ba6f2c99097edaca235907d9201ae04b394821192`.
It also matches the earlier run with native positive+negative encoding, confirming
that skipping the unused negative leaves this CFG-1 result unchanged.

The clothing reference contains two people. Native and optimized outputs retain
two people despite the prompt's single-person request. This is an optimization
regression check, not a passed instruction-following or identity quality evaluation.

Artifacts are under the development instance's
`output/qwen-image21-validation-20261020/`. Use `generation-final-v3/` and
`forward-benchmark-final.json`. Archived records also contain the withdrawn
prototype's results; the current scripts and nodes expose only retained paths.

All five downloaded Qwen files are now present in both instances' appropriate
model folders using user-authorized hard links. Only development code and
standalone development validation were changed; production services and plugin
code were not updated. Existing servers need to reload the new plugin code
before its nodes become available.
