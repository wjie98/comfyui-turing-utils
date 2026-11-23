from __future__ import annotations

from .nodes.attention import AttentionStrategy
from .nodes.bernini import BerniniContextWindowsCore, BerniniInpaintCondition
from .nodes.krea2 import Krea2IdentityEditConditioning
from .nodes.latent import SetVideoLatentNoiseMask, VideoLatentCompositeMasked
from .nodes.loaders import ConvRotCLIPLoader, ConvRotDiffusionModelLoader
from .nodes.qwen_image21 import QwenImage21Optimize, QwenImage21Conditioning
from .nodes.logic import IsInputPresent, LazyIfElse, StageBarrier, StagePath
from .nodes.media import ResizeImageIfPresent, VideoMotionContactSheet
from .nodes.sigmas import SigmasToString
from .nodes.multimodal_chat import MultimodalPromptChat
from .nodes.sec import _SeCLoader, SeCTrackVisualConcept, _SeCApply
from .nodes.minimax import (
    H3AddNoise,
    MiniMaxH3LatentUpscale,
    _H3UpscaleApply,
    _H3UpscaleLoader,
)
from .nodes.minimax_vae import (
    MiniMaxH3VideoVAEDecode,
    MiniMaxH3VideoVAEEncode,
)
from .nodes.minimax_sampler import (
    MiniMaxH3ImageKSampler,
    MiniMaxH3KSampler,
    _H3KSamplerApply,
)
from .nodes.minimax_references import (
    H3AudioReference,
    H3BuildConditioning,
    H3KeyframeReference,
    H3ImageReference,
    H3LatentInfo,
    H3SemanticReference,
    H3VideoReference,
)
from .nodes.video_sequence import (
    H3SetAudioPrefixNoiseMask,
    LoadIndexedVideoSegment,
    MergeIndexedVideoSegments,
    SaveIndexedVideoSegment,
    TrimVideoContinuationPrefix,
    VideoContinuationConcat,
    VideoPrefixContextNoise,
)
from .nodes.video_roi import (
    VideoMaskGuidedCrop,
    VideoMaskGuidedStitch,
    VideoPadForOutpaint,
)
from .nodes.visual_prompt import MaskToVisualPrompts
from .nodes.video_padding import VideoFramesPadding


NODE_CLASS_MAPPINGS = {
    "TuringUtilsVideoFramesPadding": VideoFramesPadding,
    "TuringUtilsAttentionStrategy": AttentionStrategy,
    "TuringUtilsConvRotDiffusionModelLoader": ConvRotDiffusionModelLoader,
    "TuringUtilsConvRotCLIPLoader": ConvRotCLIPLoader,
    "TuringUtilsQwenImage21Optimize": QwenImage21Optimize,
    "TuringUtilsQwenImage21Conditioning": QwenImage21Conditioning,
    "TuringUtilsSetVideoLatentNoiseMask": SetVideoLatentNoiseMask,
    "TuringUtilsVideoLatentCompositeMasked": VideoLatentCompositeMasked,
    "TuringUtilsBerniniContextWindowsCore": BerniniContextWindowsCore,
    "TuringUtilsBerniniInpaintCondition": BerniniInpaintCondition,
    "TuringUtilsKrea2IdentityEditConditioning": Krea2IdentityEditConditioning,
    "TuringUtilsIsInputPresent": IsInputPresent,
    "TuringUtilsLazyIfElse": LazyIfElse,
    "TuringUtilsStageBarrier": StageBarrier,
    "TuringUtilsStagePath": StagePath,
    "TuringUtilsSigmasToString": SigmasToString,
    "TuringUtilsH3AddNoise": H3AddNoise,
    "TuringUtilsH3LatentInfo": H3LatentInfo,
    "TuringUtilsH3KeyframeReference": H3KeyframeReference,
    "TuringUtilsH3ImageReference": H3ImageReference,
    "TuringUtilsH3VideoReference": H3VideoReference,
    "TuringUtilsH3AudioReference": H3AudioReference,
    "TuringUtilsH3SemanticReference": H3SemanticReference,
    "TuringUtilsH3BuildConditioning": H3BuildConditioning,
    "_TuringUtilsH3UpscaleLoader": _H3UpscaleLoader,
    "TuringUtilsMiniMaxH3LatentUpscale": MiniMaxH3LatentUpscale,
    "_TuringUtilsH3UpscaleApply": _H3UpscaleApply,
    "TuringUtilsMiniMaxH3KSampler": MiniMaxH3KSampler,
    "TuringUtilsMiniMaxH3ImageKSampler": MiniMaxH3ImageKSampler,
    "_TuringUtilsH3KSamplerApply": _H3KSamplerApply,
    "TuringUtilsResizeImageIfPresent": ResizeImageIfPresent,
    "TuringUtilsVideoMotionContactSheet": VideoMotionContactSheet,
    "TuringUtilsMultimodalPromptChat": MultimodalPromptChat,
    "TuringUtilsMiniMaxH3VideoVAEDecode": MiniMaxH3VideoVAEDecode,
    "TuringUtilsMiniMaxH3VideoVAEEncode": MiniMaxH3VideoVAEEncode,
    "TuringUtilsLoadIndexedVideoSegment": LoadIndexedVideoSegment,
    "TuringUtilsSaveIndexedVideoSegment": SaveIndexedVideoSegment,
    "TuringUtilsMergeIndexedVideoSegments": MergeIndexedVideoSegments,
    "TuringUtilsVideoPrefixContextNoise": VideoPrefixContextNoise,
    "TuringUtilsVideoContinuationConcat": VideoContinuationConcat,
    "TuringUtilsTrimVideoContinuationPrefix": TrimVideoContinuationPrefix,
    "TuringUtilsH3SetAudioPrefixNoiseMask": H3SetAudioPrefixNoiseMask,
    "TuringUtilsVideoMaskGuidedCrop": VideoMaskGuidedCrop,
    "TuringUtilsVideoMaskGuidedStitch": VideoMaskGuidedStitch,
    "TuringUtilsVideoPadForOutpaint": VideoPadForOutpaint,
    "TuringUtilsMaskToVisualPrompts": MaskToVisualPrompts,
    "_TuringUtilsSeCLoader": _SeCLoader,
    "TuringUtilsSeCTrackVisualConcept": SeCTrackVisualConcept,
    "_TuringUtilsSeCApply": _SeCApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TuringUtilsVideoFramesPadding": "Video Frames Padding",
    "TuringUtilsAttentionStrategy": "Configure Attention Strategy",
    "TuringUtilsConvRotDiffusionModelLoader": "Load ConvRot DiT",
    "TuringUtilsConvRotCLIPLoader": "Load ConvRot CLIP",
    "TuringUtilsQwenImage21Optimize": "Qwen Image 2.1 多图优化",
    "TuringUtilsQwenImage21Conditioning": "Qwen Image 2.1 多图条件",
    "TuringUtilsSetVideoLatentNoiseMask": "Set Video Latent Noise Mask",
    "TuringUtilsVideoLatentCompositeMasked": "Video Latent Composite Masked",
    "TuringUtilsBerniniContextWindowsCore": "Bernini Context Windows",
    "TuringUtilsBerniniInpaintCondition": "Bernini Inpaint Condition",
    "TuringUtilsKrea2IdentityEditConditioning": "Krea2 Identity Edit Conditioning",
    "TuringUtilsIsInputPresent": "Is Input Present",
    "TuringUtilsLazyIfElse": "Lazy If / Else",
    "TuringUtilsStageBarrier": "Stage Barrier",
    "TuringUtilsStagePath": "Stage Path (Internal)",
    "TuringUtilsSigmasToString": "Sigmas To String",
    "TuringUtilsH3AddNoise": "H3 Add Noise",
    "TuringUtilsH3LatentInfo": "H3 Latent Info",
    "TuringUtilsH3KeyframeReference": "H3 Keyframe Reference",
    "TuringUtilsH3ImageReference": "H3 Image Reference",
    "TuringUtilsH3VideoReference": "H3 Video Reference",
    "TuringUtilsH3AudioReference": "H3 Audio Reference",
    "TuringUtilsH3SemanticReference": "H3 Semantic Reference",
    "TuringUtilsH3BuildConditioning": "H3 Build Conditioning",
    "_TuringUtilsH3UpscaleLoader": "H3 Upscale Loader (Internal)",
    "TuringUtilsMiniMaxH3LatentUpscale": "MiniMax H3 Latent Upscale",
    "_TuringUtilsH3UpscaleApply": "MiniMax H3 Latent Upscale Apply (Internal)",
    "TuringUtilsMiniMaxH3KSampler": "MiniMax H3 KSampler",
    "TuringUtilsMiniMaxH3ImageKSampler": "MiniMax H3 Image KSampler",
    "_TuringUtilsH3KSamplerApply": "MiniMax H3 KSampler Apply (Internal)",
    "TuringUtilsResizeImageIfPresent": "Resize Image If Present",
    "TuringUtilsVideoMotionContactSheet": "Video Motion Contact Sheet (Experimental)",
    "TuringUtilsMultimodalPromptChat": "Multimodal Prompt Chat",
    "TuringUtilsMiniMaxH3VideoVAEDecode": "MiniMax H3 Video VAE Decode",
    "TuringUtilsMiniMaxH3VideoVAEEncode": "MiniMax H3 Video VAE Encode",
    "TuringUtilsLoadIndexedVideoSegment": "Load Indexed Video Segment",
    "TuringUtilsSaveIndexedVideoSegment": "Save Indexed Video Segment",
    "TuringUtilsMergeIndexedVideoSegments": "Merge Indexed Video Segments",
    "TuringUtilsVideoPrefixContextNoise": "Video Prefix Context Noise",
    "TuringUtilsVideoContinuationConcat": "Video Continuation Concat",
    "TuringUtilsTrimVideoContinuationPrefix": "Trim Video Continuation Prefix",
    "TuringUtilsH3SetAudioPrefixNoiseMask": "H3 Set Audio Prefix Noise Mask",
    "TuringUtilsVideoMaskGuidedCrop": "Video Mask Guided Crop",
    "TuringUtilsVideoMaskGuidedStitch": "Video Mask Guided Stitch",
    "TuringUtilsVideoPadForOutpaint": "Video Pad For Outpaint",
    "TuringUtilsMaskToVisualPrompts": "Mask to Visual Prompts",
    "_TuringUtilsSeCLoader": "SeC Loader (Internal)",
    "TuringUtilsSeCTrackVisualConcept": "SeC Track Visual Concept",
    "_TuringUtilsSeCApply": "SeC Track Visual Concept Apply (Internal)",
}
