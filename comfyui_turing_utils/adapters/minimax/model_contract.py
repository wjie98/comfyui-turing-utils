"""ComfyUI DiT types and the block signature used by H3 model patches."""

import inspect

from comfy.ldm.minimax import model


_BLOCK_PARAMETERS = (
    "x",
    "t_emb",
    "mod_segments",
    "rope_freqs",
    "transformer_options",
    "attention",
)


def is_h3_image_dit(diffusion_model) -> bool:
    return isinstance(diffusion_model, model.MiniMaxH3ImageModel)


def is_h3_dit(diffusion_model) -> bool:
    return isinstance(diffusion_model, model.MiniMaxH3Model)


def has_h3_block_signature(forward) -> bool:
    try:
        parameters = tuple(inspect.signature(forward).parameters)
    except (TypeError, ValueError):
        return False
    return parameters in {_BLOCK_PARAMETERS, ("self", *_BLOCK_PARAMETERS)}
