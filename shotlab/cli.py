"""CadensView CLI——给 agent / 脚本调用的命令行分析工具。

所有命令输出 JSON（stdout），错误输出 {"ok": false, "error": ...}（stderr + 非零退出码）。
项目打开自 .shotlab 文件；检测索引缓存于 ~/.cadensview-cli/index/。
任意工作目录启动：
    E:/CadensView/.venv/Scripts/python.exe -I -X utf8 E:/CadensView/shotlab/cli.py <命令>
仓库根目录也可使用 python -m shotlab.cli <命令>。

用法示例（cv 表示上述入口）：
    cv shots E:/p.shotlab
    cv stats E:/p.shotlab --shots 1,3
    cv detect E:/movie.mkv --save E:/p.shotlab
    cv scale E:/p.shotlab
    cv palette E:/p.shotlab --frame 120
    cv tags E:/p.shotlab set --shot 1 --tags 对话,夜景
    cv export csv E:/out/汇总.csv --project E:/p.shotlab --shots 1,3
    cv export sheet E:/out/长图.png --project E:/p.shotlab --columns 5
    cv export gif E:/out/s2.gif --project E:/p.shotlab --shot 2
    cv export frame E:/out/f.png --project E:/p.shotlab --frame 120
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shotlab.rpc import Session  # noqa: E402

FFMPEG = REPO_ROOT / "tools" / "ffmpeg.exe"
WEIGHTS = REPO_ROOT / "models" / "transnet-split.onnx"


class CliError(RuntimeError):
    pass


def _out(payload, pretty=False):
    indent = 2 if pretty else None
    print(json.dumps(payload, ensure_ascii=False, indent=indent))


@contextmanager
def _session(project=None):
    """Own one command's session, including cleanup when opening or work fails."""
    session = Session()
    try:
        if project is not None:
            session.open_project(project)
        yield session
    finally:
        # This only releases the store; it does not build a state snapshot.
        session.close_current_store()


def _save(session):
    if session.store is not None:
        session.settings['shot_tags'] = session.shot_tags
        session.settings['scale_labels'] = session.scale_labels
        session.store.save(session.project, session.settings)
        session.dirty = False


def _cli_index(video: str, limit=None):
    with _session() as session:
        result = session.build_index(video, limit=limit)
        index = result["index"]
        return {
            "video": result["video_path"],
            "frames": len(index["pts"]),
            "complete": index["complete"],
            "width": index["metadata"].get("width"),
            "height": index["metadata"].get("height"),
            "duration_s": round((index["metadata"].get("duration_us") or 0) / 1e6, 2),
        }


def _cli_detect(video: str, device="auto", save=None):
    """建索引（带缓存）→ 检测 → 返回镜头；可选存为项目。"""
    path = Path(video)
    if not path.is_file():
        raise CliError(f"视频不存在: {video}")
    # 索引缓存：path+size+mtime 摘要
    stat = path.stat()
    digest = hashlib.sha256(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    cache_dir = Path.home() / ".cadensview-cli" / "index"
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_json = cache_dir / f"{digest}.json"

    with _session() as session:
        if index_json.exists():
            from t0.media import FrameIndex
            session.index = FrameIndex(**json.loads(index_json.read_text(encoding="utf-8")))
            session.index.verify_source()
            session.video_path = str(path.resolve())
        else:
            session.build_index(str(path))
            index_json.write_text(json.dumps(session.index.__dict__), encoding="utf-8")

        session.new_project()
        out_dir = cache_dir / f"det-{digest}-{int(time.time())}"
        worker = subprocess.run(
            [sys.executable, "-m", "shotlab.onnx_worker",
             "--index", str(index_json), "--out", str(out_dir),
             "--weights", str(WEIGHTS), "--device", device],
            cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", timeout=3600)
        if worker.returncode == 3:
            # GPU 失败 → CPU 重试一次（协议约定）
            out_dir = cache_dir / f"det-{digest}-cpu-{int(time.time())}"
            worker = subprocess.run(
                [sys.executable, "-m", "shotlab.onnx_worker",
                 "--index", str(index_json), "--out", str(out_dir),
                 "--weights", str(WEIGHTS), "--device", "cpu"],
                cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", timeout=3600)
        if worker.returncode != 0:
            raise CliError(f"检测失败（退出码 {worker.returncode}）: {worker.stdout[-300:]}")
        state = session.apply_detection(str(out_dir))
        payload = {"shots": state["shots"], "frame_count": state["frame_count"]}
        if save:
            session.save_as(save)
            payload["project"] = session.project_path
        return payload


def _cli_export_gif(project: str, shot_number: int, out: str):
    with _session(project) as session:
        shots = [s for s in session.state()["shots"] if not s["pending"]]
        shot = next((s for s in shots if s["number"] == shot_number - 1), None)
        if shot is None:
            raise CliError(f"镜头 {shot_number} 不存在或未完成")
        index = session.index
        start_s = float(index.pts[shot["start"]] * index.tb)
        end_s = float(index.pts[shot["end"] - 1] * index.tb) + float(index.durations[shot["end"] - 1] * index.tb)
        duration = end_s - start_s
        output = subprocess.run(
            [str(FFMPEG), "-nostdin", "-loglevel", "error",
             "-ss", f"{start_s:.6f}", "-t", f"{duration:.6f}",
             "-i", index.path,
             "-vf", "fps=12,scale=480:-1:flags=lanczos,split[a][b];"
                    "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
             "-loop", "0", out],
            capture_output=True, text=True, encoding="utf-8", timeout=600)
        out_path = Path(out)
        if output.returncode != 0 or not out_path.exists():
            raise CliError(f"GIF 导出失败: {output.stderr[-300:]}")
        return {"path": str(out_path.resolve()), "size_kb": out_path.stat().st_size // 1024,
                "duration_s": round(duration, 3)}


def _cli_export_frame(project: str, frame: int, out: str):
    with _session(project) as session:
        index = session.index
        frame = max(0, min(frame, len(index.pts) - 1))
        seconds = float(index.pts[frame] * index.tb)
        output = subprocess.run(
            [str(FFMPEG), "-nostdin", "-loglevel", "error",
             "-ss", f"{seconds:.6f}", "-i", index.path,
             "-map", "0:v:0", "-an", "-frames:v", "1", out],
            capture_output=True, text=True, encoding="utf-8", timeout=120)
        out_path = Path(out)
        if output.returncode != 0 or not out_path.exists():
            raise CliError(f"截帧失败: {output.stderr[-300:]}")
        return {"path": str(out_path.resolve()), "frame": frame}


# -- 子命令表 -----------------------------------------------------------------

def _parse_shot_numbers(value):
    try:
        numbers = [int(part.strip()) for part in value.split(",")]
        if not numbers or any(number < 1 for number in numbers):
            raise ValueError()
    except ValueError:
        raise argparse.ArgumentTypeError("镜头编号须为逗号分隔的正整数，例如 1,3") from None
    return numbers


def _selection_params(session, shot_numbers):
    if shot_numbers is None:
        return {}
    by_number = {shot["number"] + 1: shot for shot in session.state()["shots"]}
    shot_ids = []
    for number in shot_numbers:
        shot = by_number.get(number)
        if shot is None:
            raise CliError(f"镜头 {number} 不存在")
        shot_ids.append(shot["id"])
    return {"shot_ids": shot_ids}


def cmd_shots(args):
    with _session(args.project) as session:
        state = session.state()
        return {"video": state["video_path"], "project": state["project_path"],
                "frame_count": state["frame_count"], "shots": state["shots"]}


def cmd_stats(args):
    with _session(args.project) as session:
        from shotlab.rpc import METHODS
        return METHODS["stats.run"](
            session, _selection_params(session, args.shots), None)


def cmd_palette(args):
    with _session(args.project) as session:
        return _palette_one(session, args.frame)


def _palette_one(session, frame):
    from shotlab.statistics import build_context
    from shotlab.statistics.colors import analyze_frame_palette
    context = build_context(session.project, session.index)
    return analyze_frame_palette(context, frame)


def cmd_scale(args):
    with _session(args.project) as session:
        result = session.scale_run()
        _save(session)
        return result


def cmd_tags(args):
    with _session(args.project) as session:
        shots = session.state()["shots"]
        if args.action == "list":
            return {"tags": {s["number"] + 1: s["tags"] for s in shots if s["tags"]}}
        by_number = {s["number"] + 1: s for s in shots}
        shot = by_number.get(args.shot)
        if shot is None:
            raise CliError(f"镜头 {args.shot} 不存在")
        if args.action == "set":
            tags = sorted({t.strip() for t in args.tags.split(",") if t.strip()})
            session.shot_tags[shot["id"]] = tags
        elif args.action == "remove":
            tags = [t for t in session.shot_tags.get(shot["id"], []) if t != args.tag]
            if tags:
                session.shot_tags[shot["id"]] = tags
            else:
                session.shot_tags.pop(shot["id"], None)
        _save(session)
        return {"shot": args.shot, "tags": session.shot_tags.get(shot["id"], [])}


def cmd_detect(args):
    return _cli_detect(args.video, device=args.device, save=args.save)


def cmd_index(args):
    return _cli_index(args.video, limit=args.limit)


def cmd_motion(args):
    with _session(args.project) as session:
        result = session.motion_run(session.analysis_scope(_selection_params(session, args.shots)))
        values = [row["motion"] for row in result["shots"] if row.get("motion") is not None]
        payload = dict(
            duration_s=result["duration_s"],
            mean_motion=sum(values) / len(values) if values else None,
            parameters=result["parameters"],
            shots=result["shots"])
        if args.spectrum:
            from shotlab.motion import write_spectrum
            payload["spectrum"] = write_spectrum(result, args.spectrum)
        return payload


def cmd_export(args):
    methods = {"csv": "export.shots_csv", "sheet": "export.contact_sheet", "pdf": "export.pdf_sheet"}
    if args.kind in methods:
        with _session(args.project) as session:
            from shotlab.rpc import METHODS
            params = {"path": args.out, **_selection_params(session, args.shots)}
            if args.kind != "csv":
                params["columns"] = args.columns
            return METHODS[methods[args.kind]](session, params, None)
    if args.kind == "slit":
        with _session(args.project) as session:
            return session.slit_run({"path": args.out, "width": args.width,
                                     "columns": args.columns,
                                     **_selection_params(session, args.shots)})
    if args.kind == "stills":
        with _session(args.project) as session:
            return session.stills_run({"path": args.out, "count": args.count, "width": args.width,
                                       **_selection_params(session, args.shots)})
    if args.kind == "motion":
        with _session(args.project) as session:
            from shotlab.motion import write_spectrum
            result = session.motion_run(session.analysis_scope(_selection_params(session, args.shots)))
            return write_spectrum(result, args.out)
    if args.shots is not None:
        raise CliError("--shots 仅适用于 csv/sheet/pdf/slit/stills/motion 导出")
    if args.kind == "gif":
        if args.shot is None or args.shot < 1:
            raise CliError("GIF 导出须指定 --shot 正整数镜头编号")
        return _cli_export_gif(args.project, args.shot, args.out)
    if args.kind == "frame":
        if args.frame is None:
            raise CliError("截帧导出须指定 --frame")
        return _cli_export_frame(args.project, args.frame, args.out)
    raise CliError(f"未知导出类型: {args.kind}")


def cmd_subtitles(args):
    with _session(args.project) as session:
        from shotlab.rpc import METHODS
        loaded = METHODS["subtitles.load"](session, {"path": args.srt}, None)
        payload = {"cues": loaded["cues"]}
        if args.at is not None:
            found = METHODS["subtitles.at"](session, {"timestamp_us": int(args.at * 1_000_000)}, None)
            payload["text"] = found["text"]
        return payload


def cmd_version(_args):
    return {"name": "cadensview-cli", "repo": str(REPO_ROOT)}


def build_parser():
    parser = argparse.ArgumentParser(prog="cv", description="CadensView 镜头分析 CLI（JSON 输出）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="建立视频帧索引")
    p.add_argument("video")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("detect", help="自动切镜检测")
    p.add_argument("video")
    p.add_argument("--device", choices=("auto", "cpu", "dml"), default="auto")
    p.add_argument("--save", default=None, help="保存为 .shotlab 项目路径")
    p.set_defaults(fn=cmd_detect)

    p = sub.add_parser("shots", help="列出项目镜头")
    p.add_argument("project")
    p.set_defaults(fn=cmd_shots)

    p = sub.add_parser("stats", help="剪辑统计（ASL/分布/密度）")
    p.add_argument("project")
    p.add_argument("--shots", type=_parse_shot_numbers, default=None,
                   help="选择镜头编号（1 起，逗号分隔）；省略则分析全部已完成镜头")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("palette", help="单帧五色调色板")
    p.add_argument("project")
    p.add_argument("--frame", type=int, required=True)
    p.set_defaults(fn=cmd_palette)

    p = sub.add_parser("motion", help="逐帧运动量与镜头平均运动量")
    p.add_argument("project")
    p.add_argument("--shots", type=_parse_shot_numbers, default=None,
                   help="选择镜头编号（1 起，逗号分隔）；省略则分析全部已完成镜头")
    p.add_argument("--spectrum", default=None, help="同时导出运动指纹 PNG")
    p.set_defaults(fn=cmd_motion)

    p = sub.add_parser("scale", help="逐镜头景别识别")
    p.add_argument("project")
    p.set_defaults(fn=cmd_scale)

    p = sub.add_parser("tags", help="标签 list|set|remove")
    p.add_argument("project")
    p.add_argument("action", choices=("list", "set", "remove"))
    p.add_argument("--shot", type=int, default=None, help="镜头编号（1 起）")
    p.add_argument("--tags", default="", help="逗号分隔标签（set）")
    p.add_argument("--tag", default=None, help="单个标签（remove）")
    p.set_defaults(fn=cmd_tags)

    p = sub.add_parser("export", help="导出 csv|sheet|pdf|slit|stills|motion|gif|frame")
    p.add_argument("kind", choices=("csv", "sheet", "pdf", "slit", "stills", "motion", "gif", "frame"))
    p.add_argument("out")
    p.add_argument("--project", required=True)
    p.add_argument("--shots", type=_parse_shot_numbers, default=None,
                   help="csv/sheet/pdf 选择镜头编号（1 起，逗号分隔）；省略则导出全部已完成镜头")
    p.add_argument("--columns", type=int, default=5, help="长图列数")
    p.add_argument("--shot", type=int, default=None, help="镜头编号（gif）")
    p.add_argument("--frame", type=int, default=None, help="帧号（frame）")
    p.add_argument("--width", type=int, default=160, help="slit 每镜头列数 / stills 宽度")
    p.add_argument("--count", type=int, default=100, help="stills 张数")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("subtitles", help="加载 SRT 字幕并查询台词")
    p.add_argument("project")
    p.add_argument("srt")
    p.add_argument("--at", type=float, default=None, help="查询某秒的台词")
    p.set_defaults(fn=cmd_subtitles)

    p = sub.add_parser("version", help="版本信息")
    p.set_defaults(fn=cmd_version)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _out(args.fn(args), pretty=args.pretty if hasattr(args, "pretty") else False)
    except CliError as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
    except Exception as error:  # noqa: BLE001 — CLI 边界统一兜底
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"},
                         ensure_ascii=False), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
