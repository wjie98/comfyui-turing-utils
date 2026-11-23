# Turing Utils 节点清单

当前开发工作树有 **43 个公开节点、6 个隐藏内部执行节点**。
右键菜单只使用 `Turing Utils/分类` 一层子目录，搜索仍可按节点名定位。

## 完整公开清单

| 分类 | 数量 | 节点 |
|---|---:|---|
| Models | 5 | Load ConvRot DiT；Load ConvRot CLIP；Configure Attention Strategy；Qwen Image 2.1 多图优化；Qwen Image 2.1 多图条件 |
| Prompt | 1 | Multimodal Prompt Chat |
| Video | 13 | Video Frames Padding；Resize Image If Present；Video Motion Contact Sheet；Load Indexed Video Segment；Save Indexed Video Segment；Merge Indexed Video Segments；Video Prefix Context Noise；Video Continuation Concat；Trim Video Continuation Prefix；H3 Set Audio Prefix Noise Mask；Video Mask Guided Crop；Video Mask Guided Stitch；Video Pad For Outpaint |
| Mask | 4 | Set Video Latent Noise Mask；Video Latent Composite Masked；Mask to Visual Prompts；SeC Track Visual Concept |
| MiniMax H3 | 13 | H3 Add Noise；H3 Latent Info；H3 Keyframe Reference；H3 Image Reference；H3 Video Reference；H3 Audio Reference；H3 Semantic Reference；H3 Build Conditioning；MiniMax H3 KSampler；MiniMax H3 Image KSampler；MiniMax H3 Latent Upscale；MiniMax H3 Video VAE Encode；MiniMax H3 Video VAE Decode |
| Bernini | 2 | Bernini Context Windows；Bernini Inpaint Condition |
| Krea2 | 1 | Krea2 Identity Edit Conditioning |
| Workflow | 4 | Is Input Present；Lazy If / Else；Stage Barrier；Sigmas To String |

H3 音频保护节点放在 Video，与延续拼接、截断工具相邻。
Sigmas To String 将原生 SIGMAS 转为可复制的十进制 STRING，正向转换复用官方 ManualSigmas。
Configure Attention Strategy 放在 Models；Sol/SLA/Veda/Spark/none 通过 DynamicCombo 选择，
只显示相关参数，参数保持普通控件。Chat 的原 Options 参数使用高级设置；
H3 KSampler 的主曲线 scheduler/manual 和单段/渐进/二采使用原生 DynamicCombo。
不引入全局配置、侧栏状态或额外模型缓存。

## 已删除的公开入口

- Image Sol、Static Virtual KV：删除节点、专用适配器和运行分支；通用映射 K/V 协议与共享 kernel 保留。
- 单独 Sol/SLA/Veda Attention 入口：改用 Configure Attention Strategy。
- Patch MiniMax H3 Block Cache (Experimental)：删除专用实现。
- 旧 Wan/H3 补帧入口：改用 Video Frames Padding 的 type。
- H3 Concat/Separate AV Latent：改用官方 Concat/Separate AV Latent，
  API 类型为 LTXVConcatAVLatent / LTXVSeparateAVLatent，官方实现支持 H3。
- Multimodal Chat Options 及 Chat 的 options 接口：改用节点内高级参数。
- 旧公开 SeC/Upscaler 加载及内部 Apply ID：删除，用户只使用已整合加载的应用节点。

不提供兼容别名或自动替换；旧工作流使用这些 ID 时需要手动调整。

## 隐藏执行单元与共享模型

普通工作流保留六个内部单元：Stage Path、SeC Loader/Apply、H3 Upscale Loader/Apply、H3 KSampler Apply。
内部单元使用原生开发者模式标记隐藏；开启开发者模式后允许显示。
显示名统一以 (Internal) 结尾，节点说明标注内部用途，公开入口保留原名。
Chat 的原 Options 参数为高级参数；H3 KSampler 的采样模式和曲线使用原生动态分支；
其余普通节点完全使用原生参数布局，
SeC/Crop 接线后的控件行为交由 ComfyUI。
H3 Keyframe Reference 固定三组输入/输出，Stage Barrier 固定八组输入/输出；
未连接的位置保持为空，不再动态增减输出。
内部单元不是正常用户入口，仅按原生开发者模式规则显示。
加载/应用使用新的私有 ID，旧加载入口不再注册。

多个 SeC 或 Upscaler/H3 KSampler 应用节点若模型加载参数相同，提交时共享同一个加载节点，
应用参数仍各自独立。模型使用 ComfyUI 正常缓存和卸载管理，不维护额外常驻权重副本。
真实 PromptExecutor 测试覆盖多个应用共享一次加载、跨执行缓存复用、
应用参数变化不重载、加载参数变化触发重载。

## 已整合和刻意保留的边界

已整合：Attention 入口、Chat 高级参数、通用补帧、SeC/放大加载界面。
高级参数仍属于节点输入，参与工作流保存和缓存失效。

参考素材与视频准备采用三个可展开子图，见
[配置与模板](node-configuration.md)。不新增重复官方 Ref2VA 的 Python 全功能节点，
也不把二采或 Upscaler 作为必要阶段。

- 参考编码与语义编码、条件构造仍可独立缓存；未使用素材不应触发加载。
- 视频准备将图像与 mask 补帧分支分开，避免修改 mask 就重做 VAE。
- AV 模板使用官方音频编码与 AV 拼接，音频保护单独设置。
- 文件读写/合并、裁剪/回贴、前缀加噪、截断、mask 获取和采样调度仍独立；
  它们具有不同执行位置、缓存与副作用。
