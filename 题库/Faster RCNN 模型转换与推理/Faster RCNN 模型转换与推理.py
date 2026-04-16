import colorsys
import json
from typing import List, Tuple
import cv2
import numpy as np
import onnx
import onnxruntime
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image

def convert(weight_path: str, save_path: str) -> None:
    model=fasterrcnn_resnet50_fpn(weights=None,weights_backbone=None)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    #构建虚拟张量:PyTorch 导出 ONNX 使用的是“追踪（Tracing）”机制。需要一次“空跑”来记录图结构
    dummy_input=torch.randn(1,3,640,640)

    #动态维度
    dynamic_axes={
        'input':{0:'batch_size',2:'height',3:'weight'},
        'boxes':{0:'batch_size'},
        'labels':{0:'batch_size'},
        'scores':{0:'batch_size'}
    }

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        input_names=['input'],
        output_names=['boxes','labels','scores'],
        dynamic_axes=dynamic_axes

    )

    #TODO

def inference(onnx_file: str, image_file: str, label_path: str) -> Tuple[np.ndarray, List[List[float]], List[str], List[float]]:
    #打开标签集合
    with open(label_path,'r',encoding='utf-8') as f:
        label_map = json.load(f)
    #载入onnx模型
    session=onnxruntime.InferenceSession(onnx_file)

    #读图并处理图像
    img=cv2.imread(image_file)
    #由于CV读图是BGR的颜色通道，需要改成RGB
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #数据预处理，归一化
    img_normalized = img_rgb.astype(np.float32)/255

    # ONNX Runtime 通常需要 (Batch, Channel, Height, Width) 的输入格式
    # 因此进行 HWC(opencv读图格式) 到 CHW 的转换，并增加一维 Batch
    input_tensor = np.transpose(img_normalized, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # 4. 运行推理
    input_name = session.get_inputs()[0].name
    outputs = session.run(['boxes', 'labels', 'scores'], {input_name: input_tensor})
    boxes = outputs[0]
    labels = outputs[1]
    scores = outputs[2]
    # 5. 格式转换与标签映射
    boxes_list = boxes.tolist() if boxes.size > 0 else []
    scores_list = scores.tolist() if scores.size > 0 else []
    class_names = [label_map.get(str(int(label)), str(int(label))) for label in labels] if labels.size > 0 else []
    return img_rgb, boxes_list, class_names, scores_list

    #TODO

def generate_palette(num_colors: int) -> List[str]:
    hsv_tuples = [(x / num_colors, 0.8, 0.9) for x in range(num_colors)]
    rgb_tuples = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_tuples]
    hex_colors = ["#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in rgb_tuples]
    return hex_colors

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))

def draw_detections(image: np.ndarray, boxes: List[List[float]], labels: List[str], scores: List[float]):
    unique_labels = list(set(labels))
    palette = generate_palette(len(unique_labels))
    label_to_color = {label: hex_to_bgr(color) for label, color in zip(unique_labels, palette)}
    for box, label, score in zip(boxes, labels, scores):
        if score < 0.5:
            continue
        
        x1, y1, x2, y2 = map(int, box)
        color = label_to_color.get(label, (128, 128, 128))
        alpha = 1 - score
        temp_image = image.copy()
        cv2.rectangle(temp_image, (x1, y1), (x2, y2), color, 2) 
        image = cv2.addWeighted(temp_image, 1 - alpha, image, alpha, 0)
        
        label_text = f"ID:{label} S:{score:.2f}"
        cv2.putText(image, label_text, (x1 + 10, y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
    cv2.imwrite("detections.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    pth_file = "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    onnx_file = "fasterrcnn_resnet50_fpn.onnx"
    image_file = "example.jpg"
    label_path = "index_to_name.json"
    convert(pth_file, onnx_file)
    image, boxes, labels, scores = inference(onnx_file, image_file, label_path)
    draw_detections(image, boxes, labels, scores)