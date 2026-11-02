"""景别识别（shot scale）：MobileNetV3 ONNX 推理。

模型来源：sssabet/Shot_Type_Classification（MIT License，MovieShots 数据集训练）。
输入 128×128 RGB → ImageNet 归一化 → 5 类 softmax。
类标按字母序：['CS', 'ECS', 'FS', 'LS', 'MS']。
"""

from pathlib import Path

import numpy as np

CLASSES = ['CS', 'ECS', 'FS', 'LS', 'MS']
LABELS_ZH = {'CS': '特写', 'ECS': '大特写', 'FS': '全景', 'LS': '远景', 'MS': '中景'}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 128

_cache = {}  # session 按路径缓存（进程内单例）


def session(model_path=None):
    """获取或创建 ONNX 推理会话。"""
    import onnxruntime as ort
    if model_path is None:
        model_path = Path(__file__).parent.parent / 'models' / 'scale-mobilenetv3.onnx'
    model_path = str(model_path)
    if model_path not in _cache:
        _cache[model_path] = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
    return _cache[model_path]


def preprocess(rgb_hwc_u8):
    """(H,W,3) uint8 RGB → (1,3,128,128) float32（ImageNet 归一化）。"""
    import av
    h, w = rgb_hwc_u8.shape[:2]
    if h == IMAGE_SIZE and w == IMAGE_SIZE:
        resized = rgb_hwc_u8
    else:
        # 与 torchvision Resize(128,128) 一致的最近邻（面积太小时差异可忽略）
        ys = (np.arange(IMAGE_SIZE) * h // IMAGE_SIZE).clip(0, h - 1)
        xs = (np.arange(IMAGE_SIZE) * w // IMAGE_SIZE).clip(0, w - 1)
        resized = rgb_hwc_u8[ys][:, xs]
    chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    chw = (chw - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return chw[None]


def classify(rgb_hwc_u8, model_path=None):
    """单帧分类 → (类标, 置信度)。"""
    sess = session(model_path)
    logits = sess.run(None, {'image': preprocess(rgb_hwc_u8)})[0][0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    best = int(np.argmax(probs))
    return CLASSES[best], float(probs[best])


def classify_shot_middle(index, start_frame, end_frame, model_path=None):
    """取镜头中点帧做景别分类（调用方保证 FrameReader 已打开）。"""
    from t0.media import FrameReader
    middle = min((start_frame + end_frame) // 2, len(index.pts) - 1)
    reader = FrameReader(index)
    try:
        frame = reader.read(middle)
        array = frame.to_ndarray(format='rgb24')
    finally:
        reader.close()
    return classify(array, model_path)
