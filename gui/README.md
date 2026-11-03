# CadensView GUI（Rust 二开版）

原版桌面应用是 PyInstaller 打包、核心编译进 GUI、源码不可改。这里用 **Rust + egui 重写整个界面**，Python 核心原样复用，全部源码可改、可重编。

## 架构

```
gui/src/main.rs     egui 界面与应用状态机（镜头列表 | 播放器 | 镜头/色彩面板；F11 独立分析页）
gui/src/player.rs   FFmpeg 播放：连续播放 = rawvideo 流（按 PTS 节流），逐帧/seek = 单次 -ss 精取
gui/src/rpc.rs      Python 边车客户端（stdin/stdout 按行 JSON-RPC，请求在后台线程）
gui/src/detect.rs   spawn `python -m shotlab.onnx_worker`，读 stdout JSON-lines，退出码 3 自动 CPU 重试
gui/src/model.rs    RPC 状态镜像、FrameIndex 视图（源时间原点、排他终点、帧号换算）
gui/src/analysis.rs 独立分析工作区（剪辑/节奏/色彩、图表回看、范围标签）
gui/src/boundary.rs 共享边界校正（输入/逐帧预览/应用或取消）
shotlab/rpc.py      新增边车：index.build / project.* / edit.* / undo / apply_detection
```

职责边界：

- **Rust** 只做视图、播放解码编排、交互。
- **Python 边车** 持有全部业务状态（Project/History/ProjectStore），所以领域规则、撤销重做、SQLite 项目格式继续由经过 36 个测试的核心负责，GUI 不重写。
- **检测** 走核心既有的子进程协议：`--index/--out/--weights/--device`，`cancel` 文件取消，`playback` 文件让路，`result.json` 回传。

## 依赖

- Rust stable（本机用 `x86_64-pc-windows-gnu` 工具链）
- MinGW binutils（`dlltool`，winget: `BrechtSanders.WinLibs.POSIX.UCRT`）
- `.venv`（仓库根，已有 shotlab 依赖）
- `tools/ffmpeg.exe`（已从系统 FFmpeg 复制；另有 `ffprobe.exe`）
- `models/transnet-split.onnx`（TransNetV2 权重，SHA256 校验）

## 构建与运行

```powershell
# 推荐：脚本负责把 MinGW 与 cargo 加进 PATH
.\gui\run.ps1            # 构建后启动
.\gui\build.ps1          # 只构建
.\gui\build.ps1 --release
```

手动构建时注意 `dlltool.exe` 必须在 PATH 上，否则 `windows-*` crates 链接会报 `error calling dlltool`。

环境变量（可选）：

| 变量 | 用途 |
|---|---|
| `CADENSVIEW_ROOT` | 仓库根目录（默认从 exe 上级目录探测） |
| `CADENSVIEW_PY` | Python 解释器（默认 `<root>/.venv/Scripts/python.exe`） |

## 使用流程

1. **打开视频** → 后台建帧索引（全量解码，进度在状态栏）
2. **检测切镜** → 子进程跑 TransNetV2，自动写入镜头列表（GPU 失败自动降级 CPU）
3. 点击镜头跳转；`▶/⏸` 连续播放、`◀1帧 / 1帧▶` 逐帧、滑块拖动定位
4. 右侧写笔记（失焦/切换镜头时提交），可在当前帧拆分、与左镜头合并
5. **保存 / 另存为** `.shotlab` 项目（SQLite，与核心同格式；剪贴板式撤销重做）

## 修改指引

- 改界面：`gui/src/main.rs` 的 panel 函数；`cargo build` 增量约 5 秒。
- 改播放行为：`gui/src/player.rs`（节流、seek、逐帧缓存）。
- 加 RPC 能力：`shotlab/rpc.py` 里 `@method("...")` 注册，Rust 侧 `self.call(...)` 调用。
- 改业务规则：直接改 `shotlab/domain.py` 等原模块，`pytest tests/` 守着不变式；GUI 无需改。

## 功能总览（P0–P2 全部完成）

| 模块 | 功能 |
|---|---|
| 播放 | 连续播放/暂停、逐帧步进、帧轨 scrub（Ctrl+滚轮缩放/滚轮平移）、局部循环、缩放平移（1–16×，1.25×/步，光标锚定）、音频（音量/静音）、SRT 字幕叠层、A/B 对照 |
| 检测 | TransNetV2 切镜（GPU DirectML/CPU 自适应、可取消、进度、失败自动 CPU 重试） |
| 编辑 | 拆分/合并、共享边界精确校正（输入帧号、逐帧预览、应用/取消）、笔记、撤销重做、保存/另存为（SQLite；默认拒绝覆盖，确认后临时文件校验再替换） |
| 缩略图 | 镜头列表行内缩略图（串行 worker+磁盘缓存+虚拟滚动） |
| 分析 | 独立工作区（F11）：剪辑 / 节奏 / 色彩 / 运动四主题，左指标右图表；柱、折线、色带可点击回看镜头与源帧；范围可选“全片已完成镜头”或“已标记镜头” |
| 运动 | 逐帧 RGB 差（阈值 10/255）→ 镜头平均运动量、逐帧曲线、最高运动镜头；可导出运动指纹 PNG（横轴源时间+切镜线） |
| 电影计量导出 | 镜头 slit-scan 扫描条网格（每镜头逐帧取中列）、范围内均匀 100 静帧（JPEG） |
| 统计 | ASL/MSL/σ/CV/百分位、时长直方图、切镜密度、滑动平均 ASL、最密集切镜窗口、快切连击、分段节奏表（编辑后自动失效重算） |
| 色彩 | 选中镜头五主色、全片秒级色带与代表帧（可取消后台任务）、当前范围镜头色板与相邻切换色差 |
| 景别 | MobileNetV3 ONNX 五类识别（特写/大特写/全景/远景/中景），列表徽章显示 |
| 标签 | 镜头标签 chips（Enter 添加、点击删除），随项目持久化，CSV 含标签列 |
| 导出 | 当前帧 PNG、镜头 GIF（palettegen 高质量）、镜头 CSV、全片分析长图、PDF 联系表（分页）；长图/PDF/静帧走后台任务，可取消 |

### 计量电影学 / 动画研究扩展

| 模块 | 功能 |
|---|---|
| 稳健统计 | MSL 为主指标并给出次序统计量 95% 区间、IQR、剪辑率（镜头/分，Poisson 95%）；ASL 降为参照（Redfern 的口径） |
| 内容区段 | 把已标记镜头标为 OP/ED/预告/回顾/眼动画/标题卡；分析页勾选“排除区段”后，统计、运动、导出都不计这些镜头 |
| 动画时间（拍数） | 与运动分析同一遍解码：保持帧判定 → 每镜头一拍一/二/三/四以上、张/秒、静止占比、变拍分段；相位相关拆出整幅平移（摇移/拖动）与作画运动；30/60 帧源提示先做 IVTC |
| 切点审校 | 审校队列按“检测置信度低 / 相邻镜头极短 / 两侧画面差异小 / 人工新增 / 检测分数次峰（疑似漏检）”排序，可逐个标“已核对”；人工改动审计给出保留/删除/新增/移动与改动比例 |
| 标注层 | カット编号（多个镜头可共用）、人工景别（1–5 快捷键，与模型/VLM 分层显示）、作品信息（片名/系列/集数/分镜/演出/作监…）；标签、カット、人工景别、区段都进撤销链 |
| 重复镜头 | 每镜头三帧感知哈希找流用镜头（相邻镜头不计）；哈希存进项目，供跨集比较找共享片段（OP/ED/回顾） |
| 多片比较 | 选多个 .shotlab：各片 MSL/剪辑率区间、两两 Mann-Whitney（Holm 校正）+ Cliff's delta、按集数趋势、按作品信息分组、共享镜头序列 |
| 一致性 | 人工景别 vs 模型、人工 vs VLM 的 Cohen's κ 与混淆矩阵 |
| 视觉大模型 | 可选：每镜头约 3 张关键帧 → 景别/运镜/角度/内容类型/是否 OP·ED·眼动画/撮影效果/一句描述；单独的“VLM 推断”层，边界改动后自动失效；可从标注一键建议区段 |
| 声音 | 只用自带 ffmpeg，一遍解码（24 分钟约 3–4 秒）：逐镜头 EBU R 128 响度（镜头内按 BS.1770 门限重新积分）、峰值瞬时响度、LRA（3 秒以上镜头）、静音占比；每个切点是否落在静音里、前后 400 ms 响度跳变、频谱变化比画面切换早（J-cut）/晚（L-cut）/同步；整体响度曲线 |
| 跨集 OP/ED | 每个项目保存 Chromaprint 音频指纹；选其他集 .shotlab 比对，找出 ≥15 秒的共同音频，按位置建议片头曲/片尾曲区段（方法参照 Jellyfin Intro Skipper，独立实现）；两边都跑过重复镜头时标出“画面也重复” |
| 研究导出 | 长表 CSV（每镜头一行，所有标注层并列）+ `.method.json`（来源、帧率/时间基、检测参数、人工改动审计、稳健统计、各标注层说明）+ `.lengths.txt`（每行一个镜头时长） |

视觉大模型支持两类服务商（配置保存在 `~/.cadensview/vlm.json`，密钥不写进项目文件）：

- **Anthropic Claude**：官方 `anthropic` SDK，默认 `claude-opus-5-5`、推理强度 low、JSON Schema 结构化输出，并开启服务端拒答回退（`fallbacks: "default"`）。可不填密钥而用 `ANTHROPIC_API_KEY`。
- **OpenAI 兼容接口**：任何接受 `image_url` data URI 的 `/chat/completions`，例如通义千问 VL（DashScope compatible-mode）、智谱 GLM-4V、本地 Ollama / LM Studio。

运行前会弹窗确认镜头数和服务商；关键帧会离开本机，版权片源请先确认服务商的数据条款。

## 主题（移植原版 Theme.qml）

设置菜单可切换 6 套配色：影院琥珀（二开版）/ 深灰 ink / 浅色 paper / 暖灰 warm /
中性纯色 graphite / 横向渐变 dusk。全部取自原版调色板，运行时即时生效。

## 快捷键（对齐原版 Main.qml）

| 键 | 功能 | 键 | 功能 |
|---|---|---|---|
| Space | 播放/暂停 | ← / → | 逐帧 |
| ↑ / ↓、Ctrl+←/→ | 上一/下一镜头 | L | 局部循环 |
| B | 显示/隐藏镜头列表 | I | 显示/隐藏笔记面板 |
| M | 静音 | + / - | 音量 ±5% |
| F | 跳到帧 | G | 跳到时间 |
| Ctrl+K | 在此帧拆分 | Ctrl+J | 与下一镜头合并 |
| Ctrl+Z / Ctrl+Shift+Z | 撤销/重做 | Ctrl+S | 保存 |
| Ctrl+E | 截帧 PNG | Ctrl+Shift+E | 分析长图 |
| Ctrl+A / Ctrl+Shift+A | 全选标记/清除标记 | Ctrl+O | 打开视频 |
| Shift/Ctrl+点击镜头 | 范围标记 / 切换单个标记 | Alt+Enter | 视频模式 |
| Ctrl+Shift+C | A/B 对照开关 | F11 | 分析工作区 |
| Esc | 关闭弹窗/退出分析或视频模式/清标记 | 1–5 / 0 | 人工景别（大特写…远景）/ 清除 |

命令行：`cadensview.exe <项目.shotlab | 视频>` 启动后直接打开（可用于文件关联）。

## 测试

```powershell
# Python 侧（含 job 协议/回归/景别/标签/导出组）
..\.venv\Scripts\python -m pytest tests\ -q

# Rust 侧（含播放帧精度夹具测试，需先生成夹具）
..\.venv\Scripts\python ..\tests\make_fixture.py %TEMP%\cv-fixture
cargo test   # 设 CV_TEST_VIDEO/CV_TEST_INDEX/CV_TEST_FFMPEG 可跑播放精度
```

## 已知限制

- 音频/字幕已接入，但长片音画漂移、暂停/seek 后的音画对齐仍缺真人长片验收；字幕仅实现 SRT，ASS 未解析
- 逐帧取帧走 ffmpeg 单次启动，约 50–150ms/次；连续播放不受影响
- 景别模型在真人电影数据（MovieShots）上训练，动画泛化未验证，置信度仅供参考；模型与训练数据条款需分别核实，不能一概称“可商用”
- 分析范围支持“全片 / 已标记镜头”并可排除已标记区段；按时间段截取（不完整镜头）尚未暴露；OP/ED 可由音频指纹、VLM 或共享镜头序列给出建议，采纳仍需人工确认
- 运动与拍数分析在不超过 320 像素宽的缩小画面上计算，长片仍需解码全部帧；取消会中止并保留项目不变
- 拍数判定的已知局限：跟拍镜头（人物不动、背景移动）会被读成每帧换画，摇移占比 ≥30% 的镜头只标“参考”；3:2 下拉的 30/60 帧源需先 IVTC；只用合成片验证过，尚无真实动画的人工时间表对照
- 声音分析只用合成音轨验证过（响度与 ffmpeg 自身的 R128 表一致到 0.1 LU 内）；静音阈值 −50 dBFS、频谱变化 6 dB、同步容差 2 帧都是起始值。J/L 判定只看频谱变化点，对白开口、音效起落同样会形成变化点，结果是供听辨的候选。指纹比对的区段首尾误差约 1 秒；没有用真实番剧多集验收过
- 审校队列的阈值（置信度 0.7、6 帧、差异 0.08、次峰 0.25）是起始值，没有用真实片标定
- 视觉大模型的标注没有用真实片做过准确率评估；Claude 后端只用模拟客户端测过请求格式，没有用真实密钥调用过
- 运动与扫描条算法参考 freder/cinemetrics，但该仓库未声明许可证，因此只借鉴方法与参数，代码为本项目独立实现；未移植章节色板、NLTK 英文字幕情感、cv2 旧版交互脚本
- `.shotlab` 项目格式与官方 exe 兼容（v1 schema + settings 扩展字段）
- 尚未逐套主题截图验收、未做所有分辨率/密集点击的完整 GUI 验收

## Agent 调用（CLI + MCP）

**CLI**（全部 JSON 输出，错误走 stderr + 非零退出码）：

```powershell
..\.venv\Scripts\python -m shotlab.cli shots   <p.shotlab>
..\.venv\Scripts\python -m shotlab.cli stats   <p.shotlab> [--shots 1,3]
..\.venv\Scripts\python -m shotlab.cli detect  <video.mkv> --save <p.shotlab> --device auto
..\.venv\Scripts\python -m shotlab.cli motion  <p.shotlab> [--shots 1,3] [--spectrum out.png]
..\.venv\Scripts\python -m shotlab.cli export slit   <out.png> --project <p.shotlab> [--shots 1,3]
..\.venv\Scripts\python -m shotlab.cli export stills <dir>     --project <p.shotlab> --count 100
..\.venv\Scripts\python -m shotlab.cli export motion <out.png> --project <p.shotlab>
..\.venv\Scripts\python -m shotlab.cli scale   <p.shotlab>
..\.venv\Scripts\python -m shotlab.cli palette <p.shotlab> --frame 120
..\.venv\Scripts\python -m shotlab.cli tags    <p.shotlab> set --shot 1 --tags 对话,夜景
..\.venv\Scripts\python -m shotlab.cli export  csv|sheet|pdf|slit|stills|motion|gif|frame --project <p.shotlab> [--shots 1,3] ...
# 研究扩展（stats/motion 也接受 --exclude-regions）
..\.venv\Scripts\python -m shotlab.cli regions  <p.shotlab> add --label OP --start 0 --end 2160
..\.venv\Scripts\python -m shotlab.cli meta     <p.shotlab> --set title=片名 --set episode=3 --set storyboard=某某
..\.venv\Scripts\python -m shotlab.cli review   <p.shotlab>
..\.venv\Scripts\python -m shotlab.cli audit    <p.shotlab>
..\.venv\Scripts\python -m shotlab.cli repeats  <p.shotlab>
..\.venv\Scripts\python -m shotlab.cli compare  ep01.shotlab ep02.shotlab ep03.shotlab --group-by storyboard --csv 比较.csv
..\.venv\Scripts\python -m shotlab.cli audio    <p.shotlab> [--cuts] [--exclude-regions]
..\.venv\Scripts\python -m shotlab.cli audio-match ep02.shotlab ep01.shotlab ep03.shotlab [--add]
..\.venv\Scripts\python -m shotlab.cli research 长表.csv --project <p.shotlab> --exclude-regions [--no-motion] [--no-audio]
..\.venv\Scripts\python -m shotlab.cli vlm-config --provider openai_compatible --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --model qwen-vl-max --api-key-env DASHSCOPE_API_KEY
..\.venv\Scripts\python -m shotlab.cli vlm      <p.shotlab> --exclude-regions
```

**MCP**（stdio，28 个工具：cv_shots/cv_stats/cv_detect/cv_palette/cv_scale/cv_tags/
cv_analysis/cv_motion/cv_motion_spectrum/cv_export_slit/cv_export_stills/
cv_export_csv/cv_contact_sheet/cv_export_pdf/cv_export_gif/cv_export_frame/cv_subtitles，
以及 cv_stats_robust/cv_regions/cv_metadata/cv_review/cv_audit/cv_repeats/cv_compare/
cv_research_export/cv_vlm/cv_audio/cv_audio_match）：

- 项目根目录 `.mcp.json` 已注册（绝对脚本 + `-I`，从任意 cwd 启动；已实测 initialize/tools.list）
- `cv_stats` / `cv_analysis` / CSV / 长图 / PDF 支持可选 `shot_ids`（取自 `cv_shots`）；省略=全部已完成镜头，空列表被拒绝
- 在支持项目级 MCP 配置的客户端里打开本仓库即自动生效；或手动把同一段配置加入客户端
