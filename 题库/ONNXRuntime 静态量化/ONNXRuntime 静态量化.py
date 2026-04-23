import os
import numpy as np
import time
from PIL import Image
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from typing import List, Optional


class CalibrationDataReaderImproved(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str) -> None:
        self.image_folder = calibration_image_folder
        self.model_path = model_path
        self.is_initialized = False
        self.data_iter: Optional[iter] = None
        self.data_size: int = 0

    def get_next(self) -> Optional[dict]:
        if not self.is_initialized:
            self._initialize()
        return next(self.data_iter, None)

    def _initialize(self) -> None:
        #TODO
        session = onnxruntime.InferenceSession(self.model_path,providers=['CPUExecutionProvider'])
        input_info = session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape
        _,C,H,W = self.input_shape

        imgs_handled = preprocess_images(self.image_folder,H,W)

        # 校准后的数据必须可以直接传入ONNX模型推理,必须转为input_seed
        img_list = [{self.input_name:img} for img in imgs_handled]

        self.data_iter = iter(img_list)
        self.data_size = len(img_list)

        self.is_initialized = True

def preprocess_images(image_folder: str, height: int, width: int) -> List[np.ndarray]:
    #TODO
    img_paths = [
        os.path.join(image_folder, f) 
        for f in os.listdir(image_folder) 
        if f.lower().endswith('.png')
    ]
    img_handled = []
    for i in img_paths:
        # 加载图片,WHC
        img0 = Image.open(i)
        # 转灰度图像WH
        img = img0.convert('L')
        # 改尺寸
        img = img.resize((width,height))
        # 归一化
        img = np.array(img).astype(np.float32) / 255.
        # 增加通道BCWH
        img_final = np.expand_dims(img,axis=(0,1))
        #img_final = np.transpose(img_final,axes=(0,1,3,2))
        img_handled.append(img_final)

    return img_handled


def static_quantization(input_model_path: str, output_model_path: str, calibration_data_path: str) -> None:
    data_reader = CalibrationDataReaderImproved(calibration_data_path, input_model_path)
    #TODO
    quantize_static(input_model_path,output_model_path,data_reader)


if __name__ == "__main__":
    input_model_path = "/home/project/mnist.onnx"
    output_model_path = "/home/project/mnist-quant.onnx"
    calibration_data_path = "/home/project/mnist"
    static_quantization(input_model_path, output_model_path, calibration_data_path)