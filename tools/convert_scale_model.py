"""一次性转换：MobileNetV3 .pt → .onnx（用 uv run --with torch 临时环境）。"""
import torch

model = torch.load('tools/scale-mobilenetv3.pt', map_location='cpu', weights_only=False)
model.eval()
dummy = torch.randn(1, 3, 128, 128)
torch.onnx.export(
    model, dummy, 'models/scale-mobilenetv3.onnx',
    input_names=['image'], output_names=['logits'],
    dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17,
)
print("ONNX exported")
# 验证
import onnxruntime as ort
session = ort.InferenceSession('models/scale-mobilenetv3.onnx', providers=['CPUExecutionProvider'])
import numpy as np
result = session.run(None, {'image': dummy.numpy()})
print(f"output shape: {result[0].shape}, classes: {result[0].shape[1]}")
assert result[0].shape == (1, 5), f"unexpected shape {result[0].shape}"
print("ONNX verified: 5-class output")
