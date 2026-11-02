"""生成带 bt709 标记的合成视频 + 索引 JSON，供 cargo 播放精度测试使用。

用法: .venv/Scripts/python tests/make_fixture.py <输出目录>
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from t0.media import build_index

out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixture")
out.mkdir(parents=True, exist_ok=True)
video = out / "clip.mkv"
with av.open(str(video), "w") as container:
    stream = container.add_stream("ffv1", rate=24)
    stream.width, stream.height, stream.pix_fmt = 160, 90, "yuv444p"
    stream.codec_context.colorspace, stream.codec_context.color_range = 1, 1
    stream.codec_context.color_primaries, stream.codec_context.color_trc = 1, 13
    for number in range(48):
        array = np.zeros((90, 160, 3), np.uint8)
        array[:] = [255, 0, 0] if number < 24 else [0, 0, 255]
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        frame.pts = number
        frame.time_base = Fraction(1, 24)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)

index = build_index(str(video))
(out / "index.json").write_text(json.dumps(index.__dict__), encoding="utf-8")
print(f"fixture ready: {video} + index.json ({len(index.pts)} frames)")
