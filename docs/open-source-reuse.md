# 开源项目与数据集复用核查

核查日期：2026-11-04。CadensView 使用 MIT 许可（`../LICENSE`）。本轮功能开发遵循「先核实格式、许可和可用性，再写代码」；下表区分**实现依据**、**可离线使用的参考数据**和**仅作对照但不可随项目分发的数据**。所有链接在核查时均可访问；外部数据集的使用条款如有变更，以原始页面为准。

## 结论与本轮实际复用方式

| 来源 | 已核实许可 / 条款 | 本轮复用 | 不复用 / 限制 |
| --- | --- | --- | --- |
| [MPI ELAN / EAF 3.0 格式说明](https://www.mpi.nl/tools/elan/EAF_Annotation_Format_3.0_and_ELAN.pdf)、[XSD](http://www.mpi.nl/tools/elan/EAFv3.0.xsd)、[ELAN 官方页](https://archive.mpi.nl/tla/elan) | ELAN 源码为 GPL v3；公开的格式规范/XSD 用于互操作 | 依据公开规范独立实现最小 EAF 3.0 导出/导入：毫秒 `TIME_SLOT`、多个 `TIER`、必填 `ANNOTATION_VALUE`、`LANGUAGE` / `CONTROLLED_VOCABULARY` / `CV_ENTRY_ML` / `CVE_VALUE` 和按 XSD `crossLinkType` 写的属性式 `CROSS_REF_LINK`；每个研究集合一个层，镜头关系作为注解间的交叉引用。完整词表、作者、裁决和版本化字段另存于 JSON 投影 | 不打包 ELAN 源码或下载到本地的 XSD/PDF 核查副本；规范 XSD 的 `REF1`/`REF2` keyref 写成子元素路径、与属性类型定义不一致，故按官方类型定义输出属性并以结构测试验证；不声称覆盖 ELAN 全部约束，也不自动授予源媒体或标注数据许可 |
| [Tangshitao/ClipShots](https://github.com/Tangshitao/ClipShots)、[标注 JSON](https://github.com/Tangshitao/ClipShots/blob/master/annotations/test.json)、[评测脚本](https://github.com/Tangshitao/ClipShots/blob/master/tools/evaluate.py)、[LICENSE](https://github.com/Tangshitao/ClipShots/blob/master/LICENSE) | 仓库代码/标注 MIT，Copyright (c) 2019 tangshitao；视频从 YouTube/微博收集，另行托管，其再分发权利未由 MIT 覆盖 | 借鉴按视频分组、`frame_num`、转换帧区间 `[begin,end)`（长度 1 为硬切、>1 为渐变）的格式，以及「预测区间与参考区间重叠才计命中」的评测口径；用于镜头切点/渐变的人工参考集适配器与单元测试 | 不下载、打包视频或整套标注；不照搬脚本，区间匹配实现独立编写；评测结果必须注明参考集与容差 |
| [AdA Film-Analytical Annotations](https://projectada.github.io/annotations/)、[Advene/AdA 手册](https://www.ada.cinepoetics.fu-berlin.de/media/ada-toolkit/Manual_Advene_AdA_Eng_Vers1_0.pdf) | 人工标注数据 CC BY-SA 3.0；本体与手册 CC BY-SA 3.0；数据含影片时间码、类型/值、作者与日期 | 借鉴「时间片段 + 类型 + 值 + 作者 + 来源」及受控词表；证据卡明确区分**记录/出处**与**判断/解释**，并保留版本元数据 | 不把 CC BY-SA 语料或本体文字直接纳入 MIT 仓库；影片本体仍受原版权约束。如果未来导入 AdA RDF，须另附署名并评估衍生作品的许可兼容性 |
| [W3C Web Annotation / Media Fragments（AdA 编码说明）](https://projectada.github.io/annotations/) | 公开标准 | 证据卡和镜头关系使用源帧 `[start,end)` 与相对秒数，外部引用可写成 `t=` 媒体片段；EAF 层与 JSON 层一一对应 | 不实现完整 RDF/SPARQL 栈 |
| [FilmColors/VIAN](https://github.com/FilmColors/VIAN)、[v0.9.6 LICENSE](https://github.com/FilmColors/VIAN/blob/v0.9.6/LICENSE.txt) | GNU GPL v3 | 仅借鉴交互概念：为色彩分析设置可复用区域与排除掩码、按镜头/阶段组织观察、把自动色值当作可人工复核的证据 | 不复制 VIAN 源码、模型或 UI 资源；本项目用 NumPy/Pillow 独立计算区域统计，避免把 GPL 代码并入 MIT 项目 |
| [Sonic Visualiser](https://github.com/sonic-visualiser/sonic-visualiser)、[COPYING](https://github.com/sonic-visualiser/sonic-visualiser/blob/master/COPYING) | GNU GPL v2 | 仅借鉴「波形与频谱按同一时间轴对齐、分层显示声学线索」的工作方式 | 不复制源码；本项目用现有 NumPy FFT 与 PCM 包络独立生成频谱图 |
| [Advene](https://www.advene.org/) | 官方称 free/open software；教程有 CC BY 4.0 / CC BY-SA 许可；具体源码许可按发行包核对 | 借鉴「类型化片段注解、包与影像分离、时间轴视图」的概念 | 不打包软件、教程或 `.azp` 示例；本轮不实现 Advene 包兼容层 |
| [Cinemetrics](https://cinemetrics.uchicago.edu/how-to-use) | 工具/数据库页面的公开使用说明；贡献数据条款需在提交前核对 | 借鉴镜头长度记录、镜头变化曲线和按影片统计的表达；跨片比较继续使用中位数与分布检验 | 不复制数据库、图表或 `.cmsx` 数据；不自动上传用户数据 |
| [MovieNet](https://movienet.github.io/)、[数据说明](https://github.com/movienet/movienet-tools/blob/master/docs/wiki/data.md) | 下载需遵守 OpenDataLab 用户协议；出于版权仅发布关键帧/特征，影片受严格条件约束；[MovieShots](https://opendatalab.com/OpenDataLab/MovieShots) 同样需查平台条款 | 仅借鉴镜头景别（ECS/CS/MS/FS/LS）与运镜类型、按镜头三帧采样的思路，作为保存查询和手动分类的术语参照 | 不下载或再分发 MovieNet/MovieShots 标注和关键帧；模型/数据集的准确率结论不能外推到动画 |
| [AnimeShooter](https://qiulu66.github.io/animeshooter/)、[Hugging Face 条款](https://huggingface.co/datasets/qiulu66/AnimeShooter) | 数据集 CC BY-NC 4.0 且仅用于学术；视频源自网络、版权归原权利人，需按 ID 自行获取；模型 Apache 2.0 | 借鉴层级结构（故事 → 镜头 → 角色/场景）及「画面证据帧 + 主体掩码」的证据表达；本轮实现不含自动主体检测 | 不打包标注、视频、掩码或模型；非商业限制使其不适合作为本项目默认数据 |
| [Condensed Movies](https://www.robots.ox.ac.uk/~vgg/data/condensed-movies/)、[仓库](https://github.com/m-bain/CondensedMovies) | 数据集 CC BY 4.0，但网页明确写着视频版权仍归原权利人；元数据/字幕/面轨约 8 MB / 10 GB，视频约 250 GB | 借鉴跨片元数据、字幕与人物链接的字段设计 | 不打包片段、字幕、面轨或特征；不能把「CC BY 4.0 数据集」误解成影片本身可自由再分发 |
| [Big Buck Bunny 许可](https://peach.blender.org/about/)、[Sintel 许可](https://durian.blender.org/sharing/) | CC BY 3.0；复用影片片段须署 `© copyright 2008, Blender Foundation / www.bigbuckbunny.org` 或 `© copyright Blender Foundation | durian.blender.org`，播放整片保留片尾字幕；商标/标志除外 | 作为真实动画/电影人工验收样本的可取得来源，用于切点、渐变、曝光、构图、声音桥的人工标注练习 | 不把影片或其截帧写入仓库；用户自行下载，并在研究导出中填写来源/版本与署名 |
| [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter)、[LICENSE](https://github.com/ina-foss/inaSpeechSegmenter/blob/master/LICENSE) | MIT，Copyright (c) 2018 Ina (David Doukhan) | 作为「语音 / 音乐 / 噪声」粗分的可选外部适配器设计依据；内置路径优先用现有 FFmpeg + NumPy 特征 | 不加入强制依赖、不打包模型权重；未安装时给出明确提示 |
| [WhisperX](https://github.com/m-bain/whisperX)、[LICENSE](https://github.com/m-bain/whisperX/blob/main/LICENSE)、[pyannote-audio](https://github.com/pyannote/pyannote-audio) | WhisperX BSD-2-Clause，Copyright (c) 2024 Max Bain；pyannote-audio MIT，模型另需接受 Hugging Face 使用条款 | 作为用户自备 ASR / 说话人分离片段的适配接口；CadensView 只在用户提供 `segments` 且明确 `model` / `model_version` / `source` / `author` 时写入独立观察结果，不调用模型，也不覆盖人工说话人—角色—画外音链接和无对白标注 | 不打包模型/权重、不把推理服务设为核心依赖；外部片段的时间区间和出处需用户自行核对并确认模型条款 |
| [bkingfilm/lapian-notes](https://github.com/bkingfilm/lapian-notes)（2026-11-16 核查仓库 MIT LICENSE） | MIT | 借鉴 AI 拉片交换协议的思路（`shotlab/lapian.py`）：分析包布局（prompt.md + schema.json + 帧目录 + 身份回带防串片）、prompt 纪律条款（反模板标题 / 事实纪律 / 时间码强制 / 枚举不译）、宽松 JSON 容错导入策略、拼图模式参数取舍（4×6、320×180、≤15 张封顶）与 intensity/valence 分离的情绪曲线建模、二段式场景深拆 | 不复制其 TypeScript 源码（Web 架构不可直接复用）；刻意不用其 1 秒盲抽帧——以本项目镜头边界 + 每镜头代表帧替代；字幕自动搜索不借鉴（动画本地方言场景无源） |

## 实施边界

1. CadensView 的交换格式是自身 JSON（schema 版本化、源帧坐标、自动/人工来源、版本与裁决记录），EAF 3.0 是与 ELAN 协作的受限投影。导出 EAF 时不丢失 JSON 原文；导入 EAF 时不覆盖人工修订、曝光表或其他项目状态。
2. 自动检测、模型建议与人工判断分别保存；颜色/声音/构图数值都标为观察或候选，不自动写成研究结论。
3. 所有导出遵守「不覆盖现有文件、不写源视频或项目文件」；引用外部图片/附件只保存路径，不复制第三方素材到项目中。EAF 的 `LICENSE` 元素写明数据权利由用户声明、源媒体权利单独处理，不自动套用 CC BY 4.0。
4. `THIRD_PARTY_NOTICES.md` 记录本轮实际借鉴的格式与署名；没有复制 GPL、CC BY-SA 或 NC 项目的源码/数据到仓库，也没有把格式核查用的 PDF/XSD、视频、帧、标注、掩码或模型权重写入仓库。若以后直接纳入此类资源，需先做逐项许可兼容审查。

## 验证来源

以上许可条款均从项目仓库的 `LICENSE` / `COPYING` 或官方项目/数据集页面核实；无法从页面明确确认的（Advene 具体源码许可、Cinemetrics 数据库再分发条款、MovieShots 下载条款）在表中标注为待核对，本轮不据此复制材料。ClipShots 的 MIT 许可只覆盖仓库内容，不自动覆盖外部网盘中的视频。
