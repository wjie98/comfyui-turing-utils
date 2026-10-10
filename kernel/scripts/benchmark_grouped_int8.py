"""Compare W6 packed contraction with staged W6 and the equivalent resident S8.

This is an operator benchmark, not H3 generation timing or a quality comparison.
Use the owning instance's CUDA environment and independently installed kernel.
"""

import argparse
import json

from benchmark_backends import _elapsed_ms
import torch
import comfyui_turing_utils_kernel as kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8193)
    parser.add_argument("--input-channels", type=int, default=5376)
    parser.add_argument("--output-channels", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    m, k, n = args.rows, args.input_channels, args.output_channels
    if k % 32 or n % 8 or min(m, k, n, args.repeats) <= 0:
        parser.error("positive dimensions, K%32=0 and N%8=0 are required")
    torch.manual_seed(91)
    activation = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
    codes = torch.randint(1, 64, (n, k), dtype=torch.int32, device="cuda")
    lower = ((codes[:, 0::2] & 15) | ((codes[:, 1::2] & 15) << 4)).to(torch.int8)
    upper_codes = codes.reshape(n, k // 4, 4) >> 4
    upper_shifts = torch.arange(4, device="cuda") * 2
    upper = (upper_codes << upper_shifts).sum(-1).to(torch.int8)
    packed = torch.cat((lower, upper), dim=1)
    scales = (torch.rand(n, k // 16, device="cuda") * 4).to(torch.float8_e4m3fn)
    dense_s8 = (
        ((codes - 32) * scales.float().repeat_interleave(16, dim=1))
        .round()
        .clamp(-127, 127)
        .to(torch.int8)
    )
    row_scale = torch.full((m,), 0.001, device="cuda")
    channel_scale = torch.full((n,), 0.001, device="cuda")
    del codes, lower, upper, upper_codes
    operations = {
        "w6_auto": lambda: kernel.turing_codebook_w4a8_linear(
            activation, packed, row_scale, scales, channel_scale
        ),
        "w6_staged": lambda: kernel.turing_codebook_w4a8_linear(
            activation, packed, row_scale, scales, channel_scale, chunk_rows=4096
        ),
        "resident_s8": lambda: kernel.turing_int8_linear(
            activation, dense_s8, row_scale, channel_scale
        ),
    }
    reference = operations["resident_s8"]()
    for name, operation in operations.items():
        torch.testing.assert_close(operation(), reference, rtol=0.008, atol=0.001)
        print(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "kernel": kernel.__version__,
                    "shape": [m, n, k],
                    "operation": name,
                    "milliseconds": _elapsed_ms(operation, 3, args.repeats),
                    "scope": "prequantized contraction; excludes activation rotation and quantization",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
