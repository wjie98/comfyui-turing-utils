"""Material boundary types shared by nodes, validation and the workspace UI."""

MATERIAL_TYPES = {
    "text": ("STRING",),
    "image": ("IMAGE",),
    "video": ("IMAGE", "AUDIO"),
    "audio": ("AUDIO",),
}
MATERIALS = {"TuringMaterial" + kind.title(): kind for kind in MATERIAL_TYPES}
