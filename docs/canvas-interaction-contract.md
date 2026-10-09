# Canvas 交互契约

- 原生工作流标签页承载项目和卡片编辑；不另开窗口、iframe 或第二套图编辑器。
- 新建、打开、编辑命令保留当前工作流。编辑上下文属于各标签页 extra，不使用全局 editing。
- 项目创建必须显式选择空目录；读取不创建项目。核心文件不合法就报错，不静默恢复为空图。
- 目录浏览限于所属实例 output。项目、模板及素材路径都由后端检查。
- 节点使用默认标题、插槽、按钮、文本框和 combo；禁止自绘卡片皮肤、端点覆盖层、缩放系统。
- 按画布标记启用菜单和队列限制，不改变普通工作流的行为。经典 LiteGraph 必须可用。
- 卡片模板和实例分离，稳定 ID 不依赖显示名称和顺序；复制一个原生节点不能冒充新实例。
- 素材文本内联保存；视频使用 VIDEO。中间结果不能直接成为跨卡片输出。
- canvas.json 是项目事实来源。前端原生图是编辑表示，保存时验证节点、插槽和连线。
- 保存使用 revision，冲突报错；不覆盖其它标签页的新结果。删除节点不隐式删除素材文件。
- 播放器按需建立，默认不自动播放；切标签、离屏、页面隐藏和移除节点时释放。
- 裁剪仅会话内有效。模型缓存沿用 ComfyUI；不能为了清理任务中间结果清空模型缓存。
- 只封装缺失的项目语义，不引入通用组件框架，不导入前端带 hash 的内部打包文件。

## 代码归属

- web/material_workspace.js：原生节点、命令、标签页、局部执行交互。
- web/lib/canvas_ports.js、endpoint_interaction.js：普通工作流端点的动态插槽。
- workspace/native.py：项目图表示及保存校验。
- workspace/store.py、workflow_files.py：素材与实例持久化。
- workspace/compiler.py、cache.py：局部计算编译及中间结果清理。
- workspace/routes.py：受限项目、模板和素材 API。

## 回归

使用 ops/test-dev.sh -q -k material_workspace 检查数据与执行边界。
tests/browser/material_native.mjs 检查经典编辑器的原生控件、新标签页、项目布局与局部执行。
修改项目或节点协议时同步增加回归；不保留依赖已删除页面的测试和兼容接口。
