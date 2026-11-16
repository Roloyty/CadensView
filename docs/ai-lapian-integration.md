# 镜头级 AI 拉片接入方案

基于现有 MCP server，把"叙事级拉片"接进来。参考项目：[lapian-notes](https://github.com/bkingfilm/lapian-notes)（MIT，借鉴点见 §8）。

> **实施状态（2026-11-16）：P1 / P2 / P3 已全部落地。**
> - 核心：`shotlab/lapian.py`（协议、包导出、容错导入、深拆、直连、长图）；narrative 为独立
>   project layer（`rpc.py` 的 `PROJECT_LAYERS`/`UNDOABLE_LAYERS`，随项目持久化、可撤销）。
> - 入口：MCP 工具 `cv_lapian_package / cv_lapian_import / cv_narrative / cv_narrative_compare /
>   cv_lapian_deep / cv_lapian_deep_import / cv_lapian_run / cv_export_lapian`；CLI `lapian-package /
>   lapian-import / lapian-narrative / lapian-compare / lapian-deep / lapian-deep-import /
>   lapian-run` 与 `export lapian`；GUI「分析 → AI 拉片」页签。
> - 测试：`tests/test_lapian.py` 14 例 + GUI 路由测试；全套通过（见下）。
> - 操作手册（含 GLM-5.3-Flash 直连配置）：`docs/lapian-agent-recipe.md`。
> - P2 的真机实测按约定留待 GLM-5.3-Flash 上手时进行；直连入口与配置路径已备好。

## 0. 现状与缺口

CadensView 已有三层能力：

| 层 | 内容 | 现状 |
|---|---|---|
| 量测层 | TransNetV2 切镜、色彩/色带、运动量、音频响度、景别 | ✅ 成熟 |
| 逐镜头语义层 | `cv_vlm`：景别/运镜/角度/内容/OP·ED/特效/一句描述 | ✅ 已有（`shotlab/vlm.py`，Anthropic 结构化输出 + openai_compatible 双后端） |
| 叙事层 | 场景划分、剧情线、段落作用、观众情绪曲线、全片结构 | ❌ 缺失 |

缺口本质：`cv_vlm` 是**逐镜头独立调用**，模型看不到全片上下文，回答不了"这场戏在全片中的作用 / 属于哪条剧情线 / 观众体验如何变化"。lapian-notes 恰好只做了这一层（但它没有镜头检测，用 1 秒盲抽凑合）。**用我们的真镜头边界 + 量测数据 + 逐镜头 VLM 标注做证据，去换它的叙事层**，是互补而不是重复造轮子。

## 1. 总体架构

一个"拉片包"协议，三条消费路径：

```
                 ┌─ A. Agent 路径（主推）：ZCode 等 MCP agent 编排
cv_lapian_package ─→ 包（prompt.md/schema.json/manifest/frames/srt）
                 ├─ B. 手动路径：包丢网页版 AI（零 API key），JSON 存文件
                 └─ C. 内置路径（后期可选）：复用 vlm.py 的 backend_for 直连
                              ↓ 返回 JSON
cv_lapian_import ─→ 容错校验 → narrative layer 写回项目
```

- A 路径零新 UI：agent 调 `cv_lapian_package` → 调模型（智谱 GLM-4.5V / Claude 等）→ `cv_lapian_import`，全程 MCP。
- B 路径照搬 lapian-notes 验证过的"免 key 手动流"，国内模型用拼图模式。
- C 路径不建议先做：vlm.py 的 provider 抽象已存在，随时能加"全片分析"入口，非必需。

## 2. 数据层：narrative project layer

与 `regions` / `vlm_labels` 同机制的独立 layer（`rpc.py` 的 `PROJECT_LAYERS`，不动 `PER_SHOT_LAYERS`）：

```python
narrative = {
    "version": 1,
    "model": "", "prompt_version": 0, "updated_at": "",   # 缓存失效，同 vlm.py 模式
    "identity": {"title": "", "source": "", "shot_count": 0},  # 导入校验防串片
    "story_lines": [{"id": "protagonist_action", "title": "", "description": ""}],
    "scenes": [{
        "id": "", "start_shot": 1, "end_shot": 23,        # 1-based 镜头序号区间
        "title": "", "type": "开场|冲突|推进|…",           # 枚举沿用 lapian SegmentType
        "function": "", "primary_line": "", "shared_lines": [],
        "techniques": "", "key_beats": "",                 # 每条带时间码
        "confidence": 0.0,
    }],
    "curve_points": [{                                     # 借鉴 lapian AudienceCurvePoint
        "time": 0.0, "intensity": 0, "valence": 0,
        "emotion_type": "", "rhythm_role": "", "title": "", "why": "",
    }],
    "macro": {"overall_structure": "", "rhythm_pattern": "", "writing_lessons": []},
}
```

**关键决策：场景锚定用镜头序号区间，不用秒/帧。** 切点是 CadensView 里最常被人工修订的东西；锚到镜头实体（number + id 双写）后，微调切点不摧毁叙事结构。lapian-notes 用 startTime/endTime 秒，导入时要 clamp，是它没有稳定镜头实体的无奈。

## 3. 新增 MCP 工具（3+1 个）

### cv_lapian_package(project, out_dir, shot_ids=None, srt_path="", mode="sheet", frames_per_shot=1, max_images=360)

导出分析包：

- `manifest.json` —— **证据核心，lapian-notes 没有的差异化**：每镜头一行，含 number/shot_id/起止秒/时长/VLM 景别·运镜·内容·一句描述（置信度）/人工标签/运动量/响度（有则带）/该镜头时间窗内的字幕文本。数据全部来自现有 `session.state()` + stats/audio/motion 结果，零新计算。
- `frames/`：
  - `mode="frames"`：每镜头代表帧原图，文件名 `shot-0007_00m12s3.jpg`（镜头号+时间码双烧进文件名，图内左上角再烧一遍）。
  - `mode="sheet"`：4×6 拼图（格 320×180、JPEG 0.8、≤15 张封顶，参数照抄 domesticAiPackage.ts），**每格是一个镜头的代表帧而非等间隔秒**——比 lapian-notes 的密度选择更贴分析对象。
- `subtitles.srt`（有则带）+ `prompt.md` + `schema.json`（协议见 §4）。

### cv_lapian_import(project, text, dry_run=False)

容错导入（防御层照抄 aiImport.ts 的实战经验）：

- payload 多层解包（segments/scenes/data 等容器字段都认）、代码围栏剥离、括号配平提取 JSON；
- 身份校验：identity 与 `metadata` / 源文件名 / shot_count 不符直接拒绝；
- **双通道定位**：`start_shot` 序号优先，缺省时按 timecode（`52:30` 解析）映射到覆盖镜头并 clamp；镜头号越界计数警告而非静默丢弃；
- `dry_run` 返回预览（场数/曲线点数/未知枚举数/越界镜头数），确认后写 narrative layer；
- 默认**不碰** shot note / tags（可选参数 `write_notes=True` 时才把 techniques 附加到对应镜头 note，加 `[AI]` 前缀）。人工笔记是研究资产，AI 结果必须可整体丢弃。

### cv_narrative(project)

读 narrative layer，输出场景表/剧情线/曲线（JSON）。供 agent 与后续 GUI 使用。

### （P2）cv_lapian_deep(project, scene_id, out_dir)

场景深拆包：该场景每镜头 2–3 帧 + 台词 + 场景粗稿（narrative 里已有的），产出逐镜头拉片正文（场景/动作/对白块，时间对齐镜头），对应 lapian 的 segment deep-dive 二段式——省 token 且可控。

## 4. 协议设计（prompt.md / schema.json 要点）

模板做成 Python 常量放 `shotlab/lapian.py`（与 vlm.py 的 SYSTEM 风格一致）。纪律条款直接吸收 lapian-notes 已踩坑验证的写法：

1. **身份回带**：filmIdentity 原样带回，导入校验（borrow: movieIdentity）。
2. **事实纪律**：人物/地名/设定无字幕或画面证据一律留空，禁止脑补。
3. **标题禁模板**：场景标题必须是具体剧情事件，"开场建立"类一律拒绝。
4. **key_beats 每条带时间码**，便于人工回看核对。
5. **枚举原样返回不翻译**（SegmentType / rhythm_role / emotion_type）。
6. **量测不服从**（新增，我们特有）：manifest 里的景别/时长/运动量是仪器数据，模型判断与之矛盾时须在 `techniques` 里说明观察依据，**不得改写量测字段含义**——这是把 VLM 幻觉锚回地面真值的关键。
7. **情绪曲线反直觉条款**整段移植 lapian：intensity 是激活度不是音量，安静对峙=高强度低效价；禁止随时间机械上升；全片 12–20 点而非每场一点；valence -100~100。
8. 无字幕时切换纯画面模式措辞（lapian 已有现成文案）。

## 5. Token / 成本预算

- 24 分钟 TV 动画 ≈ 300–500 镜头。sheet 模式 360 格封顶 ≈ 15 张拼图 + manifest（约 200–400 KB 文本）+ SRT，GLM-4.5V / gemini-flash 级单次全片粗拆可行。
- 深拆按场景分批（单场 20–60 镜头 × 2–3 帧）。
- narrative 存 model + prompt_version，重跑时只分析新增/变更镜头（同 vlm.py 失效策略）。

## 6. 分期

| 期 | 内容 | 规模 |
|---|---|---|
| P1 核心闭环 | `shotlab/lapian.py`（协议+容错导入）+ narrative layer + 3 个 MCP 工具 + CLI 子命令（cli.py 现有模式） | ~600–800 行 Python，不动 GUI/Rust |
| P2 联调 | 智谱 GLM-4.5V 实测一部片；agent 技能封装（ZCode 一句话触发全流程）；cv_lapian_deep | 视实测 |
| P3 可选 | GUI 泳道/情绪曲线展示（Rust 侧）、拉片长图导出、cv_compare 接 narrative 做跨集叙事比较 | 另立项 |

P1 不碰 Rust GUI、不碰现有检测/统计代码路径，风险面小；narrative layer 即使不接 AI，也可当人工场景划分工具用（cv_regions 已是同类先例）。

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 无字幕动画（常见） | manifest.dialogue 留空，prompt 切纯画面模式 |
| 拼图小格镜头号看错 | 文件名+图内双烧镜头号与时间码；导入按 timecode 反查校验 number，不一致计入警告 |
| 超长剧场版 | max_images 封顶 + 先 sheet 粗拆后 deep 深拆的二段式 |
| AI 结果污染人工笔记 | narrative 独立 layer，默认只读可整体丢弃；写 note 需显式 opt-in |
| 串片（导错项目） | identity 三重校验（title/source/shot_count） |

## 8. lapian-notes 借鉴清单（MIT，需登记 open-source-reuse.md）

| 借鉴点 | 来源 |
|---|---|
| 分析包协议（prompt.md+schema.json+frames+identity 回带） | `src/lib/framePackage.ts` |
| 宽松 JSON 容错导入全套策略 | `src/lib/aiImport.ts` |
| 拼图模式参数与"牺牲密度换可上传"取舍 | `src/lib/domesticAiPackage.ts` |
| prompt 纪律条款（反模板/事实纪律/时间码/枚举不译） | `framePackage.ts:298-393` |
| 情绪曲线建模（intensity/valence 分离 + 反直觉规则） | `src/types.ts:87-129` |
| 二段式深拆（segmentId 回带定位） | `framePackage.ts:98-220` |

**不借鉴**：1 秒盲抽帧（我们镜头边界更准）、web/localStorage 架构、字幕自动搜索（动画场景无源）。
