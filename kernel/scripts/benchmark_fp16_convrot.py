"""Compare FP16-storage ConvRot quantizers; run in the owning kernel environment.

Reports CUDA-graph GPU time, not Python/launch overhead or end-to-end VAE time.
The split baseline uses dense FP16 rotation plus Kitchen's row quantizer; it is
a timing reference for the retired implementation, not its exact tiny-row math.
"""

import argparse
import json
import statistics

import torch
from comfy_kitchen.backends import cuda as kitchen_cuda
from comfy_kitchen.backends._activations import apply_input_act
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation
from comfyui_turing_utils_kernel import turing_fp16_int8_convrot_quantize


def graph_time(function, repeats, batches):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = function()
    for _ in range(5):
        graph.replay()
    samples = []
    for _ in range(batches):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    return statistics.median(samples), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--batches", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or args.batches < 1:
        parser.error("repeats and batches must be positive")
    torch.manual_seed(1001)
    print(
        json.dumps({"device": torch.cuda.get_device_name(), "torch": torch.__version__})
    )
    for m, k in ((128, 2048), (2048, 2048), (8192, 2048), (2048, 8192), (8192, 8192)):
        for activation in (None, "swiglu", "rms_norm"):
            x = torch.randn(
                m,
                k * (2 if activation == "swiglu" else 1),
                device="cuda",
                dtype=torch.float16,
            )
            h = _build_hadamard(256, device=x.device, dtype=x.dtype)
            weight = (
                torch.ones(k, device=x.device, dtype=x.dtype)
                if activation == "rms_norm"
                else None
            )
            eps = 1e-6

            def split():
                activated = apply_input_act(x, activation, weight, eps)
                rotated = _rotate_activation(activated, h, 256)
                return kitchen_cuda.quantize_int8_rowwise(rotated)

            def native():
                return kitchen_cuda.quantize_int8_rowwise_convrot64(
                    x,
                    256,
                    input_act=activation,
                    input_act_weight=weight,
                    input_act_eps=eps,
                )

            def fused():
                return turing_fp16_int8_convrot_quantize(
                    x, 256, activation, weight, eps
                )

            results = {}
            for name, operation in (
                ("split", split),
                ("kitchen", native),
                ("fused_fp32", fused),
            ):
                elapsed, _ = graph_time(operation, args.repeats, args.batches)
                results[name + "_ms"] = elapsed
            print(
                json.dumps(
                    {
                        "M": m,
                        "K": k,
                        "activation": activation,
                        **results,
                        "speedup_vs_split": results["split_ms"]
                        / results["fused_fp32_ms"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
