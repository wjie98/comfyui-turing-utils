# 临时任务 API

插件把 API 挂在当前 ComfyUI 服务的 `/turing` 下，不启动额外服务，也不开新端口。
生产环境继续通过现有 Unix socket 和 SSH 转发访问。它沿用 ComfyUI 的访问边界和
`Comfy-User` 用户选择，不额外提供公开网络认证网关。

任务只在本次 ComfyUI 启动期间有效。**每次启动先删除上一次的全部 API 任务，
包括未完成任务和已完成但尚未下载的结果。** 不恢复执行，不自动重试。

## 发现服务器、节点和模型

agent 从 `GET /turing/server` 开始。它同时提供状态和简短调用帮助，无需单独的 help API：

- 原有 `session_id`、`ready`、`error`、Manager 重启字段保持不变。
- `api_version=2` 标识本插件任务协议；`versions` 返回 ComfyUI 版本和插件 Git 修订。
  导出安装或没有 Git 时 `plugin_revision=null`；未提交的源码带 `dirty` 标记。
- `links` 指向节点搜索、原生节点详情、模型类别／文件、任务、下载和详细文档。
  带 `{class_type}`、`{folder}`、`{id}` 的地址是 URL 模板，替换值需要按路径段编码。
- `usage` 提供提交格式、示例 JSON、实际上传限制、结果保留时间和清理／重启规则。
  工作流格式为 `comfy_api_prompt`，直接接收原生 API prompt 图。
- `runtime` 提供当前设备、已安装内核版本及 W8A8、Sol、SLA、Spark 和映射 K/V 的
  硬件／二进制能力检测结果。它不加载模型或运行 CUDA 测试；检测失败时仅该字段报告错误。
  版本和运行能力在进程内缓存，更新源码或依赖后应重启服务。

任务 API 未就绪时 `/turing/server` 仍返回以上信息，HTTP 状态继续为 503。
节点搜索独立于任务清理状态，仍可使用：

```http
GET /turing/nodes?query=h3&limit=20
GET /turing/nodes?category=Turing%20Utils&include_internal=true
GET /object_info/TuringUtilsMiniMaxH3KSampler
GET /models
GET /models/diffusion_models
```

`/turing/nodes` 返回 `items`、过滤后的总数 `total`、`next_cursor` 和元数据读取失败的
`errors`。每项包含 `class_type`、显示名称、分类、最多 240 字符的 `summary`、内部／废弃／
实验／远程 API 节点标志及原生 `detail_url`。它只列已注册节点，不表示模型或运行条件已满足。

搜索参数如下；未知参数或非法值返回 400：

| 参数 | 行为 |
|---|---|
| `query` | 在节点标识、显示名称、分类、说明和搜索别名中匹配全部搜索词，忽略大小写 |
| `category` | 指定分类及其子分类，忽略大小写 |
| `include_internal` | 默认 `false`；`true` 包含开发节点及标识以 `_` 开头的节点 |
| `include_deprecated` | 默认 `false`；`true` 包含废弃节点 |
| `limit` | 默认 50，范围 1–100 |
| `cursor` | 使用上一页的 `next_cursor`，其余筛选条件保持一致 |

结果按 `class_type` 排序，最后一页 `next_cursor=null`。第三方节点元数据损坏时将其列在
`errors`，其他节点仍可查询。目录不调用旧版节点的 `INPUT_TYPES`；V3 使用 ComfyUI
已有的 schema 元数据。完整参数及 tooltip 按需从 `detail_url` 读取，保留 DynamicCombo
分支与 Autogrow 重复输入，不能仅根据搜索摘要拼接工作流。

模型类别和文件继续查询原生 `/models`、`/models/{folder}`。使用返回的相对文件名，
并结合对应加载器 `/object_info/{class_type}` 中的实际输入选项选择权重。类别中也可能有
非权重目录；文件存在不能证明模型家族、量化格式、预测器或显存满足执行条件。
`runtime.support` 同样只报告底层能力，不能替代具体工作流的兼容性检查。

## 定义工作流与本地能力

API v2 执行普通 ComfyUI API prompt 图。旧 Canvas 项目若要继续使用，可在支持该格式的
历史版本中导出需要的计算图，将素材桩替换成原生加载和保存节点，再在本地注册能力。
原来的项目目录和素材由用户保管。

项目能力由本地 `turing/capabilities/<name>/` 下的 `workflow.api.json`、
`capability.json` 和 `USAGE.md` 定义。配置、素材、结果、运行记录和项目脚本都放在
项目的 `turing/` 子目录，不占用其他项目目录。参数类型、默认值、范围和节点输入绑定
保存在 capability.json；客户端按结构修改图，服务器不保存能力定义。

配套 skill 位于 [skills/turing-workflow](../skills/turing-workflow/SKILL.md)，指导 agent
从零查询节点和模型、搭建工作流，或分析用户提供的工作流并提取可调参数。普通 UI JSON
需要通过 ComfyUI 原生导出得到 API 格式；不按 widgets_values 位序猜测复杂子图或动态输入。
USAGE.md 记录输入要求、参数关系、输出行为和已验证问题；运行会保留当时的说明快照。
skill 整个目录可以独立复制，客户端只需 Python 3.11+ 和 aiohttp。自动生成的
verification.json 关联工作流／接口定义哈希、成功运行 ID、服务器版本和实际参数；
图或接口定义变化后显示 stale。此标记只证明执行成功，结果质量仍需检查并写入说明。

## 提交与下载

| 请求 | 行为 |
|---|---|
| `GET /turing/server` | 状态、版本、运行能力、接口链接及简短调用帮助 |
| `GET /turing/nodes` | 可搜索、分页的节点摘要；详情复用原生 `/object_info/{class_type}` |
| `POST /turing/jobs` | 上传完整原生图和全部素材，返回 202 和任务 ID |
| `GET /turing/jobs/{id}` | 状态、节点／步数进度、耗时与估计、错误和结果 |
| `GET /turing/jobs/{id}/outputs/{output_id}` | 下载单个结果，不重新编码 |
| `DELETE /turing/jobs/{id}` | 取消／清理，完成返回 204，正在停止或下载返回 202 |

提交为 multipart/form-data，第一个字段是 `request` JSON。参数已由客户端写入图。
`workflow` 指向工作流 JSON 字段；每个文件声明 multipart 字段名及一个或多个节点输入。
输出名称对应原生 OUTPUT_NODE，不是张量输出槽。

```json
{
  "api_version": 2,
  "workflow": "graph",
  "request_id": "ca0ba11d-22d2-4bb7-bb63-1d6474c2aa74",
  "files": {
    "source_video": {
      "part": "clip",
      "targets": [{"node_id": "12", "input": "video"}]
    }
  },
  "outputs": {"result_video": {"node_id": "81"}}
}
```

```bash
curl http://127.0.0.1:18189/turing/jobs \
  -F 'request=<request.json;type=application/json' \
  -F 'graph=@workflow.api.json;type=application/json' \
  -F 'clip=@source.mp4'
```

原生图仍是标准 ComfyUI 格式：`{"12":{"class_type":"...","inputs":{...}},...}`。
嵌套输入可在 target 中增加 `"path":["file"]`；本地参数绑定也使用同一结构，
例如 `input="sigmas", path=["denoise"]` 只修改动态分支内的 denoise。
素材流式落盘后绑定输入；默认是带 `[output]` 标记的实例内路径。
接受绝对路径的加载器可以在 target 中指定 `"path_mode":"absolute"`。绑定必须指向图中
已有输入，重复绑定会拒绝。运行前执行 ComfyUI 原生校验，任一声明输出无效则整个请求
失败，不会只执行其余有效输出。正常 prompt hooks、节点替换和原生执行队列继续生效。

支持 SaveImage、SaveAudio、SaveVideo、VHS_VideoCombine 等普通输出节点。
filename_prefix 被限定到本次任务 work 子目录；VHS 强制 save_output=true。
PreviewImage、PreviewAudio 在请求副本中分别使用原生 SaveImage、SaveAudio，以便受管清理。
原生 UI 文件描述支持图片批次、音频、视频及 VHS gifs；文字 UI 结果为 UTF-8 .txt，
其他无文件 UI 结果为 .json。只有内存数据的节点需连接普通保存节点才能下载文件。

第三方保存节点须遵守受管 filename_prefix 并报告原生文件描述。返回其他目录文件会明确
失败，不会复制或删除外部文件；第三方计算节点自行写到外部位置的副作用不在管理范围内。
不改变全局 input/output/temp 路径，不新增 API 专用节点。

`request_id` 可省略；同次启动中相同 ID 和请求元数据返回原任务，不重复入队。
去重不比较重复请求的工作流文件或素材二进制内容；修改图或素材必须使用新 ID。
已删除的 ID 不能再次执行。可提供 session_id，防止旧请求在重启后进入新会话。

请求和工作流 JSON 各最大 8 MiB，所有字段内容总和受 ComfyUI --max-upload-size 限制。
实际读取字节累计检查；连续 120 秒没有下一块数据中止上传。不支持上传断点续传，
下载中断可以重试。状态为 receiving、preparing、queued、running、succeeded、failed、deleting。
结果提供 id、name、批次 index、kind、filename、bytes、sha256 和相对下载 url。
HEAD 不表示下载完成，不触发删除。

## 进度与预计时间

状态查询增加以下字段，原有字段不变；`expires` 仍是结果保留期限。

- `queue.position`：当前下一次原生执行在等待队列中的位置，从 1 开始；
  `ahead` 包括前面的等待项和正在执行的项。尚在编译、执行中或已完成时为 `null`。
  每次 API 任务直接提交一次原生执行。
- `progress.phase`：`receiving`、`preparing`、`queued`、`executing`、`finalizing`、
  `cleanup`、`succeeded` 或 `failed`。整理结果时不再有活动计算节点。
- `progress.active_nodes`：当前运行节点的 `id`、`class_type`、可用的 `title`、
  `elapsed_seconds`，以及节点主动报告时的 `value`、`max`、`fraction`、`eta_seconds`。
  没有进度上报的节点不会伪造步数；并行异步节点可以同时出现在列表中。
- `progress.nodes`：**当前这一次原生执行**已知节点数 `known`、实际完成数 `completed`
  和命中缓存数 `cached`，实际完成数不重复计入缓存。动态扩图会增加已知节点数；
  lazy 分支可能不执行全部已知节点，因此不据此生成工作流完成百分比。
- `progress.executions_started/completed`：本任务的原生执行次数，多输出在同一次执行中处理。
  `progress.outputs` 统计已整理完成的输出节点数／总数，不是最终文件数。
- `timing`：`started_at/finished_at` 为 Unix 秒时间戳，`elapsed_seconds` 是从接收任务
  开始的总耗时，`queue_seconds` 是累计等待耗时，`execution_seconds` 包括原生执行、
  整理结果和清理耗时。完成后耗时冻结，不会随结果保留时间增长。

节点 ETA 使用最近最多 5 个单位的步速，中位数估计剩余时间；忽略首个完成单位之前的
模型加载／编译时间，至少获得两个后续步速样本且波动不超过 2 倍才显示。进度重置或
总步数变化时重新估算；长时间没有进展、原估计已过期时回到 `null`。
这是当前节点剩余时间，不是整个工作流剩余时间。

整个任务的 `timing.eta_seconds` 使用同一用户、同工作流、同参数、同素材内容及绑定的成功
记录估算。至少需要两次记录，取最近最多 5 次的中位数，明显不稳定时不显示。
`eta_scope=processing` 表示不包含未知的后续排队；只有估计可用时才有
`estimated_finish`、`eta_source=history` 和 `confidence=low`。`estimate_status` 为
`estimating`、`available`、`queued`、`overdue`、`complete` 或 `unavailable`。
数据不足、正在排队或原估计已超时时，ETA 为 `null`，成功完成时为 0。
素材内容、模型冷热状态和其他负载仍会影响实际耗时，预计时间不是完成期限。
历史只在当前进程内保留，最多 128 种规格，重启后清空；失败任务不参与学习。

进度采集独立于查询频率，按原生任务 ID 和所属 API 任务归集，不读取其他任务的
节点进度，也不改变普通画布的执行方式。此版本通过状态轮询获取，不新增 WebSocket 服务。

## Python 客户端和项目工具

客户端实现统一位于 `skills/turing-workflow/scripts/turing_client/`，仅依赖 aiohttp
和标准库；复制使用时保留整个 skill 目录。插件内原有
`comfyui_turing_utils.job_api.client`、`project` 等导入路径仍解析到这套实现。
导入不加载 ComfyUI、Torch 或模型。发现方法为 server()、nodes()、node_info()、
models()；完整节点 schema 和模型目录复用原生接口。

```python
async with Client("http://127.0.0.1:18189") as client:
    result = await client.run(
        workflow="workflow.api.json",  # 也可直接传入已渲染的 dict
        outputs={"video": {"node_id": "81"}},
        files={"source": {"path": "source.mp4", "targets": [{"node_id": "12", "input": "video"}]}},
        timeout=3600,
        on_progress=lambda state: print(state["progress"], state["timing"]),
    )
    paths = await client.download_all(result["id"], "results")
```

submit() 提交后可用 watch() 异步迭代或 wait() 等待；run() 合并提交与等待。
回调支持普通或异步函数；默认每 0.5 秒查询一次，可设置 poll_interval。
等待超时不会取消、重交或删除任务。失败／取消会抛异常。

项目工具无需客户端项目安装 ComfyUI，通过 skill 的脚本运行（也可复制到其他目录）：

```bash
python skills/turing-workflow/scripts/turing.py --project /path/to/project init --url http://127.0.0.1:18189
python skills/turing-workflow/scripts/turing.py --project /path/to/project show video-edit
python skills/turing-workflow/scripts/turing.py --project /path/to/project check video-edit --params /path/to/parameters.json --file source=source.mp4 --online
python skills/turing-workflow/scripts/turing.py --project /path/to/project run video-edit --params /path/to/parameters.json --set 'seed=42' --file source=source.mp4 --detach
python skills/turing-workflow/scripts/turing.py --project /path/to/project resume JOB_ID --timeout 3600
```

init 只创建项目下的 turing/，拒绝覆盖已有配置。show 返回参数定义和使用说明。
check 检查本地类型、绑定和文件；`--online` 只读原生节点 schema，额外检查已安装节点、
当前 combo／模型选项、连接类型、输出槽和动态分支必填参数，不上传或执行。
无法确定的动态输入返回 warnings，原生自定义校验仍发生在提交时。run 在 turing/runs/<id>/ 保存
渲染工作流、能力定义、说明快照和运行记录，下载到 turing/outputs/<id>/。
`--set NAME=JSON` 保留数值、布尔和字符串类型；`--params FILE` 从 JSON 对象读取长文本等
参数，--set 覆盖同名值。文件使用 --file NAME=PATH。`headers_env` 配置可将 header 名映射
到环境变量名，避免把认证凭据写入项目。

`run --detach` 提交后立即返回记录；`resume JOB_ID` 接续已有任务，等待、下载并清理。
未启用 detach 时 run 完成同样的全流程。远端 status 与本地 activity 分开记录，等待超时
不冒充任务失败；不确定的提交不自动重交。恢复失败时结构化错误返回 run_id／job_id。
下载成功记录先落盘再 DELETE；清理失败保留成功结果并标记 cleanup_pending，resume 可重试。
清理完成后的 status／resume 可离线读取本地结果，不依赖服务器或认证配置；resume 会重新
校验 SHA256。远端重启标记 invalidated，同一会话中不存在的任务标记 unavailable。

HTTP 错误保留原 `error` 字符串，增加 `error_details` 对象：code、phase、message，以及
适用时的 node_id、class_type、exception_type 或原生 node_errors。客户端不再将原生错误
二次序列化为字符串。CLI 成功 JSON 写 stdout，进度与错误 JSON 写 stderr；成功退出 0，
请求／执行错误退出 1，参数语法错误退出 2。API 协议仍为 v2。

下载先写 .partial，核对 bytes/SHA256 后改名，全部成功后 DELETE。
下载失败保留服务器结果并清理本地未完成文件。重试会核对并跳过已有完整文件，
不覆盖不同内容的本地同名文件。download_all(delete=False) 可保留远端结果。
服务重启后客户端通过 session_id 识别旧任务失效，不自动重新提交。

## 文件生命周期与重启

任务存放在所属实例的 `output/.turing-api-jobs/<session>/<job>/`，与普通项目隔离。
此目录由插件创建所有权标记并取得独占文件锁，不能指向符号链接或接管其他非空目录。
同一实例误启动第二个进程时，该 API 不就绪，不会清理第一个进程的任务。

执行成功后先把所有结果固化到独立 `outputs`，再删除上传、受管中间文件和工作流临时副本；直通输入和保留原音频的结果也不会依赖被清理的输入文件。
失败则删除任务文件，暂时保留不含素材内容的错误状态。原生队列中属于该 API 任务的
历史记录也会移除，不清空其他任务历史或卸载全局模型。

最终结果默认保留 3600 秒，可用启动环境变量 `TURING_API_RESULT_TTL_SECONDS` 调整。
每 60 秒检查过期和失败的清理；查询／下载不会延长保留时间。DELETE 会等待正在执行的
任务停止或现有下载关闭，再删除整个任务目录。重复 DELETE 幂等返回成功。
文件删除失败会显示 `cleanup_error` 并重试，不会伪装为已完成。

启动时在开放任务请求之前清除全部上一轮会话目录。清理失败会使 API 保持不就绪，
`GET /turing/server` 返回 503 和原因；清理范围是 API 任务目录。
任务不跨重启恢复，旧 ID 在新进程中返回 404。

`await client.restart()` 复用 Manager 的 `/manager/reboot`，只发送一次重启请求，
随后等待 `session_id` 改变且 `ready=true`。Manager 返回 403 时直接报告，不绕过其策略。
进程完全卡死、无法处理 HTTP 时仍需 SSH 等外部方式恢复。此 API 不自动安装 Manager，
不增加 systemd 服务，也不会因普通工作流报错自动重启。
