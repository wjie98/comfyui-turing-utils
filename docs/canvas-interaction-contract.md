# Canvas 交互契约

- 原生工作流标签页承载项目和卡片编辑；不另开窗口、iframe 或第二套图编辑器。
- 新建、打开、编辑命令保留当前工作流。编辑上下文属于各标签页 extra，不使用全局 editing。
- 项目创建必须显式选择空目录；读取不创建项目。核心文件不合法就报错，不静默恢复为空图。
- 目录浏览限于所属实例 output。项目、模板及素材路径都由后端检查。
- 节点使用默认标题、插槽、按钮、文本框和 combo；不自绘卡片皮肤或缩放系统。
  端点允许局部拖拽手柄、插入标记和动画，不覆盖整张节点。
- 按画布标记启用菜单和队列限制，不改变普通工作流的行为。经典 LiteGraph 必须可用。
- 卡片模板和实例分离，稳定 ID 不依赖显示名称和顺序；复制一个原生节点不能冒充新实例。
- 素材文本内联保存；视频使用 VIDEO。中间结果不能直接成为跨卡片输出。
- 输入输出端点数据口仅支持 IMAGE/VIDEO/AUDIO/STRING，位置标记不传递素材。
- Inputs 的 parameter 端口支持 INT/FLOAT/BOOLEAN/STRING/COMBO；在卡片上显示原生表单，
  不创建素材连线口。默认值和控件设置属于工作流，实例参数值属于 canvas.json，按稳定 ID 保存。
  Outputs 不接受 parameter 端口；模型、latent 等运行时对象不能作为参数持久化。
- DynamicCombo 复用原生分支控件，只导出可填写参数组，支持嵌套分支。
  各分支值属于 canvas.json，切换不丢值、不自动执行；编译时仅展开选中分支。
  整组与组内字段不能分别接线，避免覆盖。工作流执行和卡片执行使用相同的分支语义。
- 文本桩只有 text 入口，连接后仍显示文本框。媒体选择复用标准 combo/上传控件。
- canvas.json 是项目事实来源。前端原生图是编辑表示，保存时验证节点、插槽和连线。
- 保存使用 revision，冲突报错；不覆盖其它标签页的新结果。删除节点不隐式删除素材文件。
- 素材提交后自动保存布局和连线；保存队列绑定项目目录，不能读取当前标签页代替目标项目。
- 播放器按需建立，默认不自动播放；切标签、离屏、页面隐藏和移除节点时释放。
- 裁剪仅会话内有效。模型缓存沿用 ComfyUI；不能为了清理任务中间结果清空模型缓存。
- 只封装缺失的项目语义，不引入通用组件框架，不导入前端带 hash 的内部打包文件。
- 新增卡片只插入新原生节点，不重载项目、不重建已有控件和连线；会话裁剪状态保持不变。
- 项目读取在单次请求内复用 canvas.json 快照；布局保存只检查已连线卡片的接口，
  不展开执行图、不读取无连线卡片。编译执行时仍重新校验实际卡片内容。
- 展示只生成卡片描述，不展开计算图；执行只展开目标卡片。跨卡片读取已发布素材，
  不为了执行一张卡片加载所有卡片的工作流。
- 素材发布先持久化任务意图，再更新 canvas.json。中断后重试或重新打开项目可恢复；
  不覆盖之后发生的手工选择，也不重复增加选择版本。
- 缩略图服务限制并发并合并重复请求；请求取消不取消其它请求正在等待的解码。
  保存队列仅保留未完成的操作，完成后释放，不随访问过的项目数量增长。
- 原生界面只保留一套选择、连线保存和任务提交 API；不保留旧独立网页的重复入口。
- 同目录与类型的历史 combo 共享已加载的文件列表；内容变化后显式刷新。
  不缓存媒体解码张量或另存模型权重，不改变原生历史下拉的使用方式。

## 代码归属

- web/material_workspace.js：原生节点、命令、标签页、局部执行交互的入口。
- web/workspace/api.js、project_state.js、material_controls.js：请求反馈、保存队列和预览生命周期。
- web/lib/canvas_ports.js、endpoint_interaction.js：普通工作流端点的动态插槽。
- web/lib/parameter_widget.js：端点与卡片共享的原生表单控件。
- web/lib/dynamic_parameter.js、workspace/parameters.py：原生分支表单、标量校验与执行输入展开。
- web/lib/port_reorder.js：端口排序、取消和局部动画；不维护另一份图。
- web/lib/material_combo.js、directory_picker.js：原生文件下拉适配和受限目录弹窗。
- workspace/native.py：项目图表示及保存校验。
- workspace/store.py、workflow_files.py、runs.py：素材与实例持久化、可恢复发布。
- workspace/projection.py、tasks.py、compiler.py：展示描述与局部计算编译。
- workspace/execution.py、cache.py：普通节点新鲜执行及中间结果清理。
- workspace/materials.py、previews.py、directories.py：媒体读写、受限缩略图和目录操作。
- workspace/routes.py：受限项目、模板和素材 API。

## 回归

使用 ops/test-dev.sh -q -k material_workspace 检查数据与执行边界。
tests/test_canvas_dynamic_parameters.py 和 tests/browser/canvas_dynamic_parameters.mjs
检查嵌套分支、保存恢复、模板隔离、重复切换及实际分支执行。
tests/browser/material_native.mjs 检查经典编辑器的原生控件、新标签页、项目布局与局部执行。
修改项目或节点协议时同步增加回归；不保留依赖已删除页面的测试和兼容接口。
