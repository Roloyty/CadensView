# AI 拉片：Agent / CLI 操作手册

面向用 MCP 工具或 CLI 编排全流程的 agent（ZCode / Codex 等）与脚本。人肉路径见
README §7 与 [集成方案](ai-lapian-integration.md)。MCP 服务已配在项目根 `.mcp.json`
（`cadensview`），工具名一律 `cv_` 前缀；CLI 等价命令在每步末尾给出（`cv` 代指
`E:/CadensView/.venv/Scripts/python.exe -I -X utf8 E:/CadensView/shotlab/cli.py`）。

## 全片拉片：三条路径

### 路径 A：API 直连（agent 一条龙，推荐）

1. 配置 OpenAI 兼容服务商一次（密钥存本机 `~/.cadensview/vlm.json`，不入项目）：

   ```
   cv vlm-config --provider openai_compatible --base-url https://open.bigmodel.cn/api/paas/v4 \
      --model glm-5.3-flash --api-key-env ZHIPU_API_KEY
   ```

   GLM-5.3-Flash 走智谱开放平台端点；换 Kimi/通义/本地 llava 只改 base-url 与 model。

2. 一条命令完成全片分析 + 写回（单条多模态消息：任务说明 + 镜头证据表 + 字幕 + 拼图，
   成功即导入叙事层）：

   ```
   cv lapian-run E:/p.shotlab
   cv_lapian_run(project="E:/p.shotlab", api_key_env="ZHIPU_API_KEY", model="glm-5.3-flash")
   ```

   `no_import=True` 只取原文人工审；`max_images` 控制拼图上限（默认 360 格）。
   先 `dry_run` 想确认质量：先 `no_import` 拿原文，再 `lapian-import --dry-run`。

### 路径 B：网页版大模型（零 API key）

1. `cv_lapian_package(project, out_dir, mode="sheets")` → 生成分析目录（prompt.md /
   schema.json / manifest.json / sheets/ / subtitles.srt）。
2. 把目录压 ZIP 上传给网页版 ChatGPT / GLM / Kimi，开场白用返回的 `chat_opener`
   原文；国内不支持 ZIP 的用散文件上传（拼图本身是散 JPG）。
3. 模型回复存成 `reply.json`（围栏、前后杂文都能容错），导回：

   ```
   cv lapian-import E:/p.shotlab reply.json --dry-run   # 先看预览与警告
   cv lapian-import E:/p.shotlab reply.json --model glm-5.3-flash
   cv lapian-import E:/p.shotlab reply.json --write-notes  # 可选：场景标题写进镜头笔记
   ```

### 路径 C：GUI

**分析 → AI 拉片** 页签：导出分析包 / 导入 AI 结果 / 导出拉片长图，场景表可一键定位。

## 场景深拆（二段式，省 token）

```
cv lapian-deep E:/deep-out --project E:/p.shotlab --scene-index 0   # 或 --scene-id
# 把 deep-out 交给模型（prompt/scene/manifest/frames 齐备），回复：
cv lapian-deep-import E:/p.shotlab reply-deep.json --scene-index 0
```

产出 `shot_cards`（逐镜头动作/台词/手法）存进叙事层对应场景。

## 读取与呈现

- `cv lapian-narrative E:/p.shotlab` / `cv_narrative(project)` — 场景/剧情线/曲线/macro。
- `cv lapian-compare E:/p1.shotlab E:/p2.shotlab E:/p3.shotlab` /
  `cv_narrative_compare(project, others)` — 跨片叙事比较：场景数/场长（镜头数）、
  类型分布、曲线峰值/低谷数、平均强度与效价。
- `cv export lapian 拉片长图.png --project E:/p.shotlab --width 1600` — 情绪曲线 +
  剧情线泳道 + 场景表 PNG（CLI 注意显式给 `--width`，共享参数默认 160 是给 slit 的）。

## Agent 行为守则

1. **先 dry-run 再写回**：`lapian.import` 的 `dry_run` 返回覆盖/越界/未知枚举警告，
   有 `scene范围越界` 或 `影片名不一致` 时先向用户报告再继续。
2. **串片防护**：导入校验 `source + shot_count`，报错即停，不要试图"修一下"结果。
3. **量测是仪器数据**：模型与 manifest 矛盾时让它在 techniques 里给画面证据，agent
   不要替它改写量测字段。
4. **切点修订后**：narrative 按镜头号锚定仍有效；若导出后镜头数变了（重检测/合并），
   shot_count 校验会拒绝旧结果——重新跑该片的包。
5. **隐私**：路径 A/B 都会把拼图关键帧发给所配服务商；版权片源先确认条款（GUI 里有同款警告）。
6. **撤销**：一次导入 = 一条撤销记录（GUI Ctrl+Z；`edit.undo`）。

## 术语对照（给不熟 CadensView 的 agent）

| 概念 | 含义 |
|---|---|
| `.shotlab` | 项目文件（SQLite）：索引 + 镜头切分 + 全部标注层 |
| 镜头 number | 1 起；`cv_shots` 列出 id/起止帧/标签/景别/VLM |
| 叙事层 narrative | 独立 project layer：scenes（镜头区间锚定）/story_lines/curve_points/macro |
| scenes.start_shot/end_shot | 1 起、含端点、首尾相接覆盖全部镜头 |
| curve_points | intensity 0-100 激活度、valence ±100 情绪方向、rhythm_role/emotion_type 枚举 |
