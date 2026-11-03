"""CadensView MCP 服务器——把镜头分析能力暴露为 MCP 工具。

运行：E:/CadensView/.venv/Scripts/python.exe -m shotlab.mcp_server（stdio 传输）
注册：项目根目录 .mcp.json 已配置；或客户端里手动指向该命令。

工具均返回 JSON 字符串；长耗时操作（detect/scale）同步完成。
项目状态按 .shotlab 文件打开——每次调用独立会话，改动的命令会写回文件。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shotlab.rpc import Session  # noqa: E402
from shotlab.cli import (  # noqa: E402
    CliError,
    _cli_detect,
    _cli_export_frame,
    _cli_export_gif,
)

mcp = FastMCP("cadensview", instructions=(
    "CadensView：动画/电影镜头分析。打开 .shotlab 项目或直接分析视频文件，"
    "提供自动切镜、剪辑统计（ASL/分布/密度）、调色板、全片色带、景别识别、"
    "镜头标签与多种导出。所有输出为 JSON。"
))


def _open(project: str) -> Session:
    session = Session()
    session.open_project(project)
    return session


def _dump(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def cv_shots(project: str) -> str:
    """列出项目全部镜头（编号/起止帧/时长/标签/景别/笔记）。"""
    try:
        state = _open(project).state()
        return _dump({"video": state["video_path"], "frame_count": state["frame_count"],
                      "shots": state["shots"]})
    except Exception as error:  # noqa: BLE001 — MCP 边界统一兜底
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_stats(project: str) -> str:
    """剪辑统计：镜头数、ASL/MSL、标准差、CV、时长直方图、切镜密度曲线。"""
    try:
        session = _open(project)
        from shotlab.statistics import build_context
        from shotlab.statistics.shots import shot_statistics
        context = build_context(session.project, session.index)
        return _dump(shot_statistics(context))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_detect(video: str, device: str = "auto", save_to: str = "") -> str:
    """对视频文件自动切镜。device=auto|cpu|dml；save_to 给定则保存为 .shotlab 项目。"""
    try:
        return _dump(_cli_detect(video, device=device, save_to=save_to or None))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_palette(project: str, frame: int) -> str:
    """单帧调色板：五种主色（hex + 占比）。"""
    try:
        session = _open(project)
        from shotlab.statistics import build_context
        from shotlab.statistics.colors import analyze_frame_palette
        context = build_context(session.project, session.index)
        return _dump(analyze_frame_palette(context, frame))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_scale(project: str) -> str:
    """逐镜头景别识别（特写/大特写/全景/远景/中景 + 置信度），结果写回项目。"""
    try:
        session = _open(project)
        result = session.scale_run()
        if session.store is not None:
            session.settings['shot_tags'] = session.shot_tags
            session.settings['scale_labels'] = session.scale_labels
            session.store.save(session.project, session.settings)
        return _dump(result)
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_tags(project: str, action: str, shot: int = 0, tags: str = "", tag: str = "") -> str:
    """镜头标签管理。action=list（词表+各镜头标签）| set（shot+tags 逗号分隔）| remove（shot+tag）。"""
    try:
        session = _open(project)
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
        if session.store is not None:
            session.settings['shot_tags'] = session.shot_tags
            session.store.save(session.project, session.settings)
        return _dump({"shot": shot, "tags": session.shot_tags.get(target["id"], [])})
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_export_csv(project: str, out_path: str) -> str:
    """镜头汇总 CSV（utf-8-sig；含标签列）。"""
    try:
        session = _open(project)
        from shotlab.rpc import METHODS
        return _dump(METHODS["export.shots_csv"](session, {"path": out_path}, None))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_contact_sheet(project: str, out_path: str, columns: int = 5) -> str:
    """全镜头网格分析长图 PNG（编号+时长+标签）。"""
    try:
        session = _open(project)
        from shotlab.rpc import METHODS
        return _dump(METHODS["export.contact_sheet"](
            session, {"path": out_path, "columns": columns}, None))
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
def cv_export_pdf(project: str, out_path: str, columns: int = 4) -> str:
    """全镜头 PDF 联系表（分页，编号+时长+标签+景别）。"""
    try:
        session = _open(project)
        from shotlab.rpc import METHODS
        return _dump(METHODS["export.pdf_sheet"](
            session, {"path": out_path, "columns": columns}, None))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


@mcp.tool()
def cv_subtitles(project: str, srt_path: str, at_seconds: float = -1.0) -> str:
    """加载 SRT 字幕；at_seconds>=0 时同时返回该时刻台词。"""
    try:
        session = _open(project)
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
def cv_analysis(project: str) -> str:
    """全片色彩分析（秒级色带样本 + 镜头色板 + 代表帧），耗时与视频长度成正比。"""
    try:
        session = _open(project)
        result = session.analysis_run()
        return _dump(result)
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    mcp.run()
