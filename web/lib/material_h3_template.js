// Ordinary editable nodes; no separate H3 execution engine.
export function createH3Template(app) {
  const created = [];
  const add = (type, title, values = {}, fields = []) => {
    const node = LiteGraph.createNode(type);
    if (!node) throw Error(`Missing node: ${type}`);
    app.graph.add(node);
    created.push(node);
    node.title = title;
    node.pos = [
      ((created.length - 1) % 5) * 350,
      Math.floor((created.length - 1) / 5) * 340,
    ];
    node.properties.materialFields = fields;
    for (const [name, value] of Object.entries(values)) {
      const widget = node.widgets?.find((w) => w.name === name);
      if (!widget) throw Error(`Missing parameter ${type}.${name}`);
      widget.value = value;
      widget.callback?.(value);
    }
    return node;
  };
  const link = (source, output, target, input) => {
    let index = target.inputs?.findIndex((i) => i.name === input);
    if (index < 0) {
      const widget = target.widgets?.find((w) => w.name === input);
      if (widget && target.convertWidgetToInput)
        target.convertWidgetToInput(widget);
      index = target.inputs?.findIndex((i) => i.name === input);
    }
    if (!(index >= 0)) throw Error(`Missing input ${target.type}.${input}`);
    source.connect(output, target, index);
  };
  try {
    const user = add("TuringMaterialText", "用户提示词", {
      prefix: "user_prompt",
      text: "一个女人在花园里说话。",
    });
    const chat = add(
      "TuringUtilsMultimodalPromptChat",
      "生成模型提示词",
      {
        base_url: "http://127.0.0.1:9200",
        system_prompt:
          "Rewrite the user's intent into a precise video generation prompt. Preserve reference numbering. Return only the prompt.",
      },
      ["base_url", "model", "system_prompt"],
    );
    const prompt = add("TuringMaterialText", "模型提示词", {
      prefix: "model_prompt",
    });
    link(user, 0, chat, "prompt");
    link(chat, 0, prompt, "value");
    const fallback = add(
      "TuringUtilsIsInputPresent",
      "模型提示词为空时使用用户提示词",
    );
    link(prompt, 0, fallback, "value");
    link(user, 0, fallback, "fallback");
    const model = add("TuringUtilsConvRotDiffusionModelLoader", "DiT", {}, [
      "unet_name",
      "patch_attention",
    ]);
    const shift = add(
      "MiniMaxH3SigmaShift",
      "Sigma shift",
      { shift_video: 12, shift_audio: 6 },
      ["shift_video", "shift_audio"],
    );
    link(model, 0, shift, "model");
    const strategy = add("TuringUtilsAttentionStrategy", "Attention", {}, [
      "strategy",
    ]);
    link(shift, 0, strategy, "model");
    const clip = add(
      "TuringUtilsConvRotCLIPLoader",
      "Text encoder",
      { type: "minimax" },
      ["clip_name"],
    );
    const vae = add("VAELoader", "Video VAE", {}, ["vae_name"]);
    const audioVae = add("VAELoader", "Audio VAE", {}, ["vae_name"]);
    const prepare = add("_TuringMaterialH3Prepare", "目标视频 / 前缀准备", {}, [
      "duration",
      "width",
      "height",
      "preserve_audio",
    ]);
    link(vae, 0, prepare, "vae");
    link(audioVae, 0, prepare, "audio_vae");
    const semantic = add(
      "TuringUtilsH3SemanticReference",
      "Semantic reference",
    );
    link(clip, 0, semantic, "clip");
    link(fallback, 1, semantic, "prompt");
    const conditioning = add("TuringUtilsH3BuildConditioning", "Conditioning");
    link(semantic, 0, conditioning, "semantic_reference");
    link(prepare, 0, conditioning, "latent");
    const keyframes = add("TuringUtilsH3KeyframeReference", "首尾帧参考");
    link(vae, 0, keyframes, "vae");
    link(prepare, 0, keyframes, "latent");
    for (const destination of [semantic, conditioning]) {
      link(keyframes, 0, destination, "first_frame");
      link(keyframes, 1, destination, "last_frame");
    }
    const imageRefs = add("TuringUtilsH3ImageReference", "图片参考");
    link(vae, 0, imageRefs, "vae");
    const videoRefs = add("TuringUtilsH3VideoReference", "视频参考");
    link(vae, 0, videoRefs, "video_vae");
    link(audioVae, 0, videoRefs, "audio_vae");
    const audioRefs = add("TuringUtilsH3AudioReference", "音频参考");
    link(audioVae, 0, audioRefs, "audio_vae");
    for (const destination of [semantic, conditioning]) {
      link(imageRefs, 0, destination, "image_reference");
      link(videoRefs, 0, destination, "video_reference");
      link(audioRefs, 0, destination, "audio_reference");
    }
    const guider = add("BasicGuider", "Guider");
    link(strategy, 0, guider, "model");
    link(conditioning, 0, guider, "conditioning");
    const scheduler = add(
      "BasicScheduler",
      "Sampling",
      { scheduler: "simple", steps: 8, denoise: 1 },
      ["scheduler", "steps", "denoise"],
    );
    link(shift, 0, scheduler, "model");
    const refiner = add(
      "_TuringMaterialH3SigmaRefiner",
      "H3 Sigma Refiner",
      {},
      ["enabled"],
    );
    link(scheduler, 0, refiner, "sigmas");
    const sampler = add(
      "KSamplerSelect",
      "Sampler",
      { sampler_name: "euler" },
      ["sampler_name"],
    );
    const noise = add("RandomNoise", "Noise", {}, ["noise_seed"]);
    const sample = add("SamplerCustomAdvanced", "Sample");
    link(noise, 0, sample, "noise");
    link(guider, 0, sample, "guider");
    link(sampler, 0, sample, "sampler");
    link(refiner, 0, sample, "sigmas");
    link(prepare, 0, sample, "latent_image");
    const separate = add("LTXVSeparateAVLatent", "Separate AV");
    link(sample, 0, separate, "av_latent");
    const decode = add("TuringUtilsMiniMaxH3VideoVAEDecode", "Decode video");
    link(separate, 0, decode, "samples");
    link(vae, 0, decode, "vae");
    const decodeAudio = add("VAEDecodeAudio", "Decode audio");
    link(separate, 1, decodeAudio, "samples");
    link(audioVae, 0, decodeAudio, "vae");
    const finish = add("_TuringMaterialH3Finish", "裁回目标帧数与音频");
    link(decode, 0, finish, "images");
    link(decodeAudio, 0, finish, "audio");
    link(prepare, 1, finish, "length");
    link(prepare, 2, finish, "prefix");
    link(prepare, 3, finish, "original_audio");
    const result = add("TuringMaterialVideo", "视频结果", { prefix: "h3" });
    link(finish, 0, result, "value");
    link(finish, 1, result, "audio");
    app.graph.setDirtyCanvas(true, true);
    return created;
  } catch (error) {
    // Only roll back nodes created by this command, never the user's existing graph.
    for (const node of created)
      if (node.graph === app.graph) app.graph.remove(node);
    throw error;
  }
}
