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
def cv_analysis(project: str, shot_ids: list[str] | None = None) -> str:
    """色彩分析（秒级色带+镜头色板+代表帧）。shot_ids 省略为全部已完成镜头，空列表无效。"""
    try:
        return _dump(_rpc(project, "analysis.run", _shot_params(shot_ids)))
    except Exception as error:  # noqa: BLE001
        return _dump({"ok": False, "error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    mcp.run(transport="stdio")
