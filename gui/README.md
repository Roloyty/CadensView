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

## 已知限制（P0 范围）

- 无音频、无字幕（分析模式静音播放）
- 无缩略图条、无 A/B 对照、无统计/色彩面板、无景别识别——按计划 P1/P2 迭代
- 逐帧取帧走 ffmpeg 单次启动，约 50–150ms/次；连续播放不受影响
- `.shotlab` 项目与官方 exe 兼容（仍为 v1 schema）
