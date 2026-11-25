import torch

from comfyui_turing_utils.adapters.jasna import _align_latents


def test_jasna_overlap_excludes_the_next_windows_first_latent():
    left = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16, 1, 1)
    right = left + 100
    _align_latents([left, right], 104, initial=True)
    torch.testing.assert_close(
        right.flatten()[:4], torch.tensor([100.0, 14.0, 15.0, 103.0])
    )

    right.add_(9)
    _align_latents([left, right], 104)
    torch.testing.assert_close(left.flatten()[-3:], torch.tensor([13.0, 17.0, 21.0]))
    torch.testing.assert_close(
        right.flatten()[:4], torch.tensor([109.0, 17.0, 21.0, 112.0])
    )
