"""CadensView MCP 服务器——把镜头分析能力暴露为 MCP 工具。

任意工作目录运行（stdio 传输）：
    E:/CadensView/.venv/Scripts/python.exe -I -X utf8 E:/CadensView/shotlab/mcp_server.py
项目根目录 .mcp.json 使用同一绝对脚本入口。

工具均返回 JSON 字符串；长耗时操作（detect/scale）同步完成。
项目状态按 .shotlab 文件打开——每次调用独立会话，改动的命令会写回文件。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from shotlab.cli import (  # noqa: E402
    CliError,
    _cli_detect,
    _cli_export_frame,
    _cli_export_gif,
    _save,
    _session,
)

mcp = FastMCP("cadensview", instructions=(
    "CadensView：动画/电影镜头分析。打开 .shotlab 项目或直接分析视频文件，"
    "提供自动切镜、剪辑统计（ASL/分布/密度）、调色板、全片色带、景别识别、"
    "镜头标签与多种导出。所有输出为 JSON。"
))


def _rpc(project: str, method: str, params: dict):
    with _session(project) as session:
        from shotlab.rpc import METHODS
        return METHODS[method](session, params, None)


def _shot_params(shot_ids):
    # Preserve explicit [] so RPC can reject it instead of silently selecting all.
    return {} if shot_ids is None else {"shot_ids": shot_ids}


def _dump(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def cv_shots(project: str) -> str:
    """列出项目全部镜头（编号/起止帧/时长/标签/景别/笔记），id 可用于 shot_ids 选择。"""
    try:
        with _session(project) as session:
            state = session.state()
            return _dump({"video": state["video_path"], "frame_count": state["frame_count"],
                          "shots": state["shots"]})
    except Exception as error:  # noqa: BLE001 — MCP 边界统一兜底
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_stats(project: str, shot_ids: list[str] | None = None) -> str:
    """剪辑统计：ASL/MSL、分布、密度。shot_ids 取自 cv_shots；省略为全部已完成镜头，空列表无效。"""
    try:
        return _dump(_rpc(project, "stats.run", _shot_params(shot_ids)))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_detect(video: str, device: Literal["auto", "cpu", "dml"] = "auto", save_to: str = "") -> str:
    """对视频自动切镜。device=auto|cpu|dml；save_to 给定则存为新 .shotlab 项目，已有文件会拒绝覆盖。"""
    try:
        return _dump(_cli_detect(video, device=device, save=save_to or None))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_palette(project: str, frame: int) -> str:
    """单帧调色板：五种主色（hex + 占比）。"""
    try:
        return _dump(_rpc(project, "palette.get", {"frame": frame}))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_scale(project: str) -> str:
    """逐镜头景别识别（特写/大特写/全景/远景/中景 + 置信度），结果写回项目。"""
    try:
        with _session(project) as session:
            result = session.scale_run()
            _save(session)
            return _dump(result)
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_tags(project: str, action: str, shot: int = 0, tags: str = "", tag: str = "") -> str:
    """镜头标签管理。action=list（词表+各镜头标签）| set（shot+tags 逗号分隔）| remove（shot+tag）。"""
    try:
        with _session(project) as session:
            shots = session.state()["shots"]
            if action == "list":
                return _dump({"tags": {s["number"] + 1: s["tags"] for s in shots if s["tags"]},
                              "scale": {s["number"] + 1: s["scale"] for s in shots if s.get("scale")}})
            by_number = {s["number"] + 1: s for s in shots}
            target = by_number.get(shot)
            if target is None:
                raise CliError(f"镜头 {shot} 不存在")
            if action == "set":
                session.shot_tags[target["id"]] = sorted({t.strip() for t in tags.split(",") if t.strip()})
            elif action == "remove":
                remaining = [t for t in session.shot_tags.get(target["id"], []) if t != tag]
                if remaining:
                    session.shot_tags[target["id"]] = remaining
                else:
                    session.shot_tags.pop(target["id"], None)
            else:
                raise CliError(f"未知 action: {action}")
            _save(session)
            return _dump({"shot": shot, "tags": session.shot_tags.get(target["id"], [])})
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_export_csv(project: str, out_path: str, shot_ids: list[str] | None = None) -> str:
    """镜头汇总 CSV（utf-8-sig，含标签）。shot_ids 省略为全部已完成镜头，空列表无效。"""
    try:
        return _dump(_rpc(project, "export.shots_csv", {"path": out_path, **_shot_params(shot_ids)}))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_contact_sheet(project: str, out_path: str, columns: int = 5,
                     shot_ids: list[str] | None = None) -> str:
    """镜头网格长图 PNG（编号+时长+标签）。shot_ids 省略为全部已完成镜头，空列表无效。"""
    try:
        return _dump(_rpc(project, "export.contact_sheet", {
            "path": out_path, "columns": columns, **_shot_params(shot_ids)}))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_export_gif(project: str, shot_number: int, out_path: str) -> str:
    """把某个镜头导出为高质量 GIF（palettegen 双通道）。"""
    try:
        return _dump(_cli_export_gif(project, shot_number, out_path))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_export_frame(project: str, frame: int, out_path: str) -> str:
    """导出指定帧为 PNG。"""
    try:
        return _dump(_cli_export_frame(project, frame, out_path))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_export_pdf(project: str, out_path: str, columns: int = 4,
                  shot_ids: list[str] | None = None) -> str:
    """PDF 联系表（分页，编号+时长+标签+景别）。shot_ids 省略为全部已完成镜头，空列表无效。"""
    try:
        return _dump(_rpc(project, "export.pdf_sheet", {
            "path": out_path, "columns": columns, **_shot_params(shot_ids)}))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_subtitles(project: str, srt_path: str, at_seconds: float = -1.0) -> str:
    """加载 SRT 字幕；at_seconds>=0 时同时返回该时刻台词。"""
    try:
        with _session(project) as session:
            from shotlab.rpc import METHODS
            loaded = METHODS["subtitles.load"](session, {"path": srt_path}, None)
            payload = {"cues": loaded["cues"]}
            if at_seconds >= 0:
                found = METHODS["subtitles.at"](
                    session, {"timestamp_us": int(at_seconds * 1_000_000)}, None)
                payload["text"] = found["text"]
            return _dump(payload)
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_motion(project: str, shot_ids: list[str] | None = None, spectrum_path: str = "") -> str:
    """逐帧运动量（RGB 差>10/255）与镜头平均运动量；可选导出运动指纹 PNG。"""
    try:
        with _session(project) as session:
            result = session.motion_run(session.analysis_scope(_shot_params(shot_ids)))
            values = [row["motion"] for row in result["shots"] if row.get("motion") is not None]
            payload = dict(duration_s=result["duration_s"],
                           mean_motion=sum(values) / len(values) if values else None,
                           parameters=result["parameters"], shots=result["shots"])
            if spectrum_path:
                from shotlab.motion import write_spectrum
                payload["spectrum"] = write_spectrum(result, spectrum_path)
            return _dump(payload)
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_motion_spectrum(project: str, out_path: str, shot_ids: list[str] | None = None) -> str:
    """运动指纹 PNG（横轴源时间，纵轴运动量；含切镜点）。"""
    try:
        with _session(project) as session:
            from shotlab.motion import write_spectrum
            result = session.motion_run(session.analysis_scope(_shot_params(shot_ids)))
            return _dump(write_spectrum(result, out_path))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_export_slit(project: str, out_path: str, width: int = 160, columns: int = 6,
                   shot_ids: list[str] | None = None) -> str:
    """镜头 slit-scan 扫描条网格 PNG（每镜头逐帧取中列）。shot_ids 省略为全部已完成镜头。"""
    try:
        with _session(project) as session:
            return _dump(session.slit_run({"path": out_path, "width": width,
                                           "columns": columns, **_shot_params(shot_ids)}))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_export_stills(project: str, out_dir: str, count: int = 100, width: int = 480,
                     shot_ids: list[str] | None = None) -> str:
    """范围内均匀抽取 count 张静帧（JPEG）。shot_ids 省略为全部已完成镜头。"""
    try:
        with _session(project) as session:
            return _dump(session.stills_run({"path": out_dir, "count": count,
                                             "width": width, **_shot_params(shot_ids)}))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_analysis(project: str, shot_ids: list[str] | None = None) -> str:
    """色彩分析（秒级色带+镜头色板+代表帧）。shot_ids 省略为全部已完成镜头，空列表无效。"""
    try:
        return _dump(_rpc(project, "analysis.run", _shot_params(shot_ids)))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


def _rpc_save(project: str, method: str, params: dict):
    """Run one RPC method on the project and write the project back."""
    with _session(project) as session:
        from shotlab.rpc import METHODS
        result = METHODS[method](session, params, None)
        _save(session)
        return result


def _error(error):
    return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_review(project: str) -> str:
    """切点审校队列：低置信度/极短镜头/两侧差异小/人工新增/疑似漏检，按优先级排序。"""
    try:
        with _session(project) as session:
            from shotlab.rpc import METHODS
            return _dump(METHODS["review.run"](session, {}, None))
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_audit(project: str) -> str:
    """检测结果与当前人工修订后的切点对比：保留/删除/新增/移动及人工改动比例。"""
    try:
        return _dump(_rpc(project, "research.audit", {}))
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_regions(project: str, action: Literal["list", "add", "remove"] = "list",
               label: str = "OP", start_frame: int = 0, end_frame: int = 0, region_id: str = "") -> str:
    """内容区段（OP/ED/preview/recap/eyecatch/title/other）。统计时 exclude_regions=true 可排除。"""
    try:
        if action == "list":
            with _session(project) as session:
                return _dump({"regions": session.regions})
        if action == "add":
            state = _rpc_save(project, "regions.add",
                              {"label": label, "start_frame": start_frame, "end_frame": end_frame})
        else:
            state = _rpc_save(project, "regions.remove", {"id": region_id})
        return _dump({"regions": state["regions"]})
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_metadata(project: str, values: dict[str, str] | None = None) -> str:
    """作品信息（title/series/episode/year/director/storyboard/episode_director/animation_director/studio/source/notes）。
    values 省略为读取；给定则合并写入（空字符串删除该字段）。"""
    try:
        if not values:
            return _dump(_rpc(project, "metadata.get", {}))
        return _dump({"metadata": _rpc_save(project, "metadata.set", {"values": values})["metadata"]})
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_stats_robust(project: str, shot_ids: list[str] | None = None, exclude_regions: bool = False) -> str:
    """稳健统计：MSL 与 95% 区间、四分位距、剪辑率（每分钟镜头数，Poisson 95%）。"""
    try:
        stats = _rpc(project, "stats.run", {**_shot_params(shot_ids), "exclude_regions": exclude_regions})
        keys = ("shot_count", "scope_duration_s", "msl_s", "msl_ci95_s", "iqr_s", "qcd", "asl_s",
                "cutting_rate_per_min", "cutting_rate_ci95_per_min")
        return _dump({k: stats["overview"].get(k) for k in keys} | {"warnings": stats["metadata"]["warnings"]})
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_repeats(project: str) -> str:
    """重复镜头（流用/bank）检测：感知哈希比对，结果写回项目以便跨集比较。"""
    try:
        with _session(project) as session:
            from shotlab.rpc import _repeats_run
            result = _repeats_run(session, {})
            _save(session)
            return _dump(result)
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_compare(projects: list[str], exclude_regions: bool = True, group_by: str = "", csv_path: str = "",
               inference_unit: str = "shot") -> str:
    """多片/多集比较：各片 MSL/剪辑率区间，两两 Mann-Whitney（Holm 校正）与 Cliff's delta，
    按集数的趋势，以及共享镜头序列（OP/ED，需先 cv_repeats）。group_by 为作品信息字段名。
    inference_unit 可选 shot、scene、film、episode；每侧样本不足时只给描述统计。"""
    try:
        from shotlab.compare import compare_projects, write_compare_csv
        result = compare_projects(projects, exclude_regions=exclude_regions, group_by=group_by or None,
                                  inference_unit=inference_unit)
        if csv_path:
            result["csv"] = write_compare_csv(result, csv_path)
        return _dump(result)
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_research_export(project: str, out_path: str, exclude_regions: bool = False,
                       shot_ids: list[str] | None = None) -> str:
    """研究长表 CSV（每镜头一行，含カット编号/区段/人工与模型景别/VLM/拍数），
    另写 .method.json（来源、检测参数、人工改动审计、稳健统计）与 .lengths.txt。"""
    try:
        with _session(project) as session:
            from shotlab.rpc import METHODS
            params = {"path": out_path, "exclude_regions": exclude_regions, **_shot_params(shot_ids)}
            session.motion_run(session.analysis_scope(params))  # 拍数列需要运动分析
            try:
                session.audio_run(session.analysis_scope(params))  # 响度/静音/声画列
            except ValueError as error:
                if "音轨" not in str(error):
                    raise
            result = METHODS["research.export"](session, params, None)
            _save(session)
            return _dump(result)
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_audio(project: str, shot_ids: list[str] | None = None, exclude_regions: bool = False,
             include_cuts: bool = False) -> str:
    """声音统计（只用 ffmpeg）：逐镜头 EBU R128 响度、LRA、静音占比；整体响度与静音占比；
    切点处是否落在静音里、前后 400 ms 响度跳变、声音变化早于（J）/晚于（L）/同步于画面切换。
    音频指纹写回项目，供 cv_audio_match 跨集比对。"""
    try:
        with _session(project) as session:
            params = dict(_shot_params(shot_ids), exclude_regions=exclude_regions)
            result = session.audio_run(session.analysis_scope(params))
            _save(session)
            payload = {k: result[k] for k in ("summary", "warnings", "offset_s", "parameters", "shots")}
            if include_cuts:
                payload["cuts"] = result["cuts"]
            return _dump(payload)
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_audio_match(project: str, others: list[str], add_regions: bool = False) -> str:
    """与其他集（.shotlab）比对 Chromaprint 音频指纹，找共同的 OP/ED 等长段音频，给出区段建议；
    若两边都跑过 cv_repeats，会标出画面上也重复的建议。add_regions=True 时直接加为区段。"""
    try:
        with _session(project) as session:
            from shotlab.rpc import METHODS, _audio_match
            result = _audio_match(session, {"paths": others})
            if add_regions:
                for item in result["suggestions"]:
                    METHODS["regions.add"](session, {"label": item["label"], "start_frame": item["start_frame"],
                                                     "end_frame": item["end_frame"]}, None)
            _save(session)
            return _dump(result)
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_vlm(project: str, shot_ids: list[str] | None = None, force: bool = False) -> str:
    """用视觉大模型标注镜头（景别/运镜/角度/内容/OP·ED/撮影效果/一句描述），结果写回项目。
    注意：会把每个镜头的关键帧发送到 ~/.cadensview/vlm.json 配置的服务商。"""
    try:
        with _session(project) as session:
            from shotlab.rpc import _vlm_run
            result = _vlm_run(session, {**_shot_params(shot_ids), "force": force})
            _save(session)
            return _dump(result)
    except Exception as error:  # noqa: BLE001
        return _error(error)


# -- AI 拉片：分析包 / 叙事层 / 深拆 / 直连 / 长图 --------------------------------


def _lapian_params(shot_ids, exclude_regions=False):
    params = _shot_params(shot_ids)
    if exclude_regions:
        params["exclude_regions"] = True
    return params


@mcp.tool()
def cv_lapian_package(project: str, out_dir: str, mode: str = "sheets", max_images: int = 360,
                      shot_ids: list[str] | None = None, exclude_regions: bool = False) -> str:
    """导出 AI 拉片分析包（prompt.md + schema.json + manifest.json 镜头量测证据 + 拼图或代表帧 + 字幕）。
    mode=sheets 拼图省流量 / frames 每镜头一图；max_images 图像数上限。把包交给任意多模态大模型，
    用返回 chat_opener 那句话开场，收到的 JSON 用 cv_lapian_import 导回。"""
    try:
        with _session(project) as session:
            from shotlab.lapian import export_package
            return _dump(export_package(session, {
                "out_dir": out_dir, "mode": mode, "max_images": max_images,
                **_lapian_params(shot_ids, exclude_regions)}))
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_lapian_import(project: str, text: str = "", path: str = "", dry_run: bool = False,
                     write_notes: bool = False, model: str = "") -> str:
    """导入大模型返回的拉片 JSON（text 直接给 / path 读文件）。写回叙事层（场景/剧情线/情绪曲线），
    可整体撤销。dry_run 只给预览（场数/警告/越界统计）；write_notes 同时把场景标题写进镜头笔记。"""
    try:
        with _session(project) as session:
            from shotlab.lapian import import_analysis
            result = import_analysis(session, {"text": text, "path": path, "dry_run": dry_run,
                                               "write_notes": write_notes, "model": model})
            if not dry_run:
                _save(session)
            return _dump(result)
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_narrative(project: str) -> str:
    """读取叙事层：场景（镜头区间锚定）/剧情线/情绪曲线/全片结构。空则各列表为空。"""
    try:
        with _session(project) as session:
            from shotlab.lapian import data
            return _dump(data(session))
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_narrative_compare(project: str, others: list[str]) -> str:
    """跨片叙事比较：当前项目 + others 各 .shotlab 的场景数/场长(镜头数)/类型分布/曲线特征。
    只读其他项目文件；没有叙事层的项目会标 has_narrative=false。"""
    try:
        with _session(project) as session:
            from shotlab.lapian import narrative_compare
            return _dump(narrative_compare(session, {"others": others}))
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_lapian_deep(project: str, out_dir: str, scene_id: str = "", scene_index: int = -1) -> str:
    """场景深拆包：该场景每镜头一图（480 宽）+ 台词 + 场景粗稿。scene_index 从 0 起，或给 scene_id。
    返回 JSON 用 cv_lapian_deep_import 导回（逐镜头卡片存进叙事层）。"""
    try:
        with _session(project) as session:
            from shotlab.lapian import export_deep
            params = {"out_dir": out_dir}
            if scene_id:
                params["scene_id"] = scene_id
            if scene_index >= 0:
                params["scene_index"] = scene_index
            return _dump(export_deep(session, params))
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_lapian_deep_import(project: str, text: str = "", path: str = "", scene_index: int = -1) -> str:
    """导入场景深拆结果（shot_cards 逐镜头卡 + 场景字段更新），写回叙事层。"""
    try:
        with _session(project) as session:
            from shotlab.lapian import deep_import
            params = {"text": text, "path": path}
            if scene_index >= 0:
                params["scene_index"] = scene_index
            result = deep_import(session, params)
            _save(session)
            return _dump(result)
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_lapian_run(project: str, base_url: str = "", model: str = "", api_key: str = "",
                  api_key_env: str = "", max_images: int = 360, timeout_s: int = 600,
                  no_import: bool = False) -> str:
    """API 直连全片拉片（OpenAI 兼容 /chat/completions，单条多模态消息：prompt+镜头表+字幕+拼图），
    成功后自动导入叙事层。base_url/model/api_key 缺省取 vlm-config 的 openai_compatible 配置；
    api_key 建议走 api_key_env 环境变量。no_import=True 只返回原文。"""
    try:
        with _session(project) as session:
            from shotlab.lapian import run_direct
            result = run_direct(session, {"base_url": base_url, "model": model, "api_key": api_key,
                                          "api_key_env": api_key_env, "max_images": max_images,
                                          "timeout_s": timeout_s, "no_import": no_import})
            if not no_import and result.get("imported"):
                _save(session)
            return _dump(result)
    except Exception as error:  # noqa: BLE001
        return _error(error)


@mcp.tool()
def cv_export_lapian(project: str, out_path: str, width: int = 1600) -> str:
    """拉片长图 PNG：情绪曲线 + 剧情线泳道 + 场景表。需要先导入叙事层。"""
    try:
        with _session(project) as session:
            from shotlab.lapian import draw_sheet
            return _dump(draw_sheet(session, {"path": out_path, "width": width}))
    except Exception as error:  # noqa: BLE001
        return _error(error)


if __name__ == "__main__":
    mcp.run(transport="stdio")
