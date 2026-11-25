"""ComfyUI entry point for Jasna LTX Restore."""

from __future__ import annotations

from comfy_api.latest import io


class JasnaLTXRestore(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsJasnaLTXRestore",
            display_name="Jasna LTX Restore",
            category="Turing Utils/Video",
            description=(
                "Restore a video crop using the schedule embedded in its Jasna DiT. "
                "Connect Load ConvRot DiT and Load VAE. "
                "The output keeps pixels where the mask is zero."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Mask.Input("mask"),
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images, mask, model, vae, seed):
        from ..adapters.jasna import restore

        return io.NodeOutput(restore(images, mask, model, vae, seed))
