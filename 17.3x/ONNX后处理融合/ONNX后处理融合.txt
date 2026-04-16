import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort

def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=10, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def postprocess(logits: torch.Tensor, num_classes: int) -> torch.Tensor:

    idx = torch.argmax(logits, dim=1)
    onehot = F.one_hot(idx, num_classes=num_classes).float()
    return onehot

def convert_to_onnx_with_postprocess(model: nn.Module, save_path: str, num_classes: int):
    model.eval()

    original_forward = model.forward
    model.forward = partial(
        export_forward,
        base_forward=original_forward,
        num_classes=num_classes
    )
    
    features = list(model.parameters())[0].shape[1]
    dummy_input = torch.randn(1, features)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        )
    model.forward = original_forward

    #TODO


def run_onnx_inference(onnx_path: str, input_data: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(onnx_path)
    
    # 2. 获取模型期望的输入名称
    input_name = session.get_inputs()[0].name
    
    # 3. 运行推理 (以字典形式喂入数据)
    outputs = session.run(None, {input_name: input_data})
    
    # 4. session.run 返回的是一个列表，提取第一个输出结果即可
    return outputs[0]
    #TODO

if __name__ == "__main__":
    set_seed()

    num_classes = 3
    model = MLPClassifier()
    
    convert_to_onnx_with_postprocess(model, "/home/project/fused_model.onnx", num_classes)

    x = np.random.randn(3, 10).astype(np.float32)
    logits = model(torch.tensor(x))
    raw_result = postprocess(logits, num_classes)
    result = run_onnx_inference("/home/project/fused_model.onnx", x)
    print(f"logits 推理结果：{logits.detach().numpy()}\n后处理结果：{raw_result.detach().numpy()}\nONNX 推理结果：{result}")