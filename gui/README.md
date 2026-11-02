# CadensView GUI（Rust 二开版）

原版桌面应用是 PyInstaller 打包、核心编译进 GUI、源码不可改。这里用 **Rust + egui 重写整个界面**，Python 核心原样复用，全部源码可改、可重编。

## 架构

```
gui/src/main.rs     egui 界面与应用状态机（三栏：镜头列表 | 播放器 | 笔记）
gui/src/player.rs   FFmpeg 播放：连续播放 = rawvideo 流（按 PTS 节流），逐帧/seek = 单次 -ss 精取
gui/src/rpc.rs      Python 边车客户端（stdin/stdout 按行 JSON-RPC，请求在后台线程）
gui/src/detect.rs   spawn `python -m shotlab.onnx_worker`，读 stdout JSON-lines，退出码 3 自动 CPU 重试
gui/src/model.rs    RPC 状态镜像、FrameIndex 视图（帧号 ↔ 时间）
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
| 播放 | 连续播放/暂停、逐帧步进、滑块跳转（PTS 精确）、A/B 对照（并排+叠加混合） |
| 检测 | TransNetV2 切镜（GPU DirectML/CPU 自适应、可取消、进度、失败自动 CPU 重试） |
| 编辑 | 拆分/合并/移边界、笔记、撤销重做、保存/另存为（SQLite，官方 exe 兼容） |
| 缩略图 | 镜头列表行内缩略图（串行 worker+磁盘缓存+虚拟滚动） |
| 统计 | ASL/MSL/σ/CV/百分位、时长直方图、切镜密度曲线（编辑后自动失效重算） |
| 色彩 | 镜头五主色调色板、全片秒级色带（可取消的后台分析） |
| 景别 | MobileNetV3 ONNX 五类识别（特写/大特写/全景/远景/中景），列表徽章显示 |
| 标签 | 镜头标签 chips（Enter 添加、点击删除），随项目持久化，CSV 含标签列 |
| 导出 | 当前帧 PNG、镜头 GIF（palettegen 高质量）、镜头 CSV、全片分析长图 |

## 测试

```powershell
# Python 侧（含 job 协议/回归/景别/标签/导出组）
..\.venv\Scripts\python -m pytest tests\ -q

# Rust 侧（含播放帧精度夹具测试，需先生成夹具）
..\.venv\Scripts\python ..	ests\make_fixture.py %TEMP%\cv-fixture
cargo test   # 设 CV_TEST_VIDEO/CV_TEST_INDEX/CV_TEST_FFMPEG 可跑播放精度
```

## 已知限制

- 无音频、无字幕（分析模式静音播放）
- 逐帧取帧走 ffmpeg 单次启动，约 50–150ms/次；连续播放不受影响
- 景别模型在真人电影数据（MovieShots）上训练，动画泛化未验证，置信度仅供参考
- `.shotlab` 项目格式与官方 exe 兼容（v1 schema + settings 扩展字段）
