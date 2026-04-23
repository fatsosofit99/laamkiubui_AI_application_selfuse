from typing import Dict, Optional, Tuple
import numpy as np
import onnx
from onnx import numpy_helper

class ONNXWeightComparator:
    def __init__(self, original_model_path: str, quantized_model_path: str) -> None:
        self.original_model = onnx.load(original_model_path)
        self.quantized_model = onnx.load(quantized_model_path)
        self.original_weights = self._extract_weights(self.original_model)
        self.quantized_weights = self._extract_weights(self.quantized_model)

    def _extract_weights(self, model: onnx.ModelProto) -> Dict[str, np.ndarray]:
        #TODO
        dick={}
        for init in model.graph.initializer:
            name=init.name
            weights = numpy_helper.to_array(init)
            dick[name]=weights
        return dick
    def _get_original_weight(self, layer_name: str) -> Optional[np.ndarray]:
        return self.original_weights.get(layer_name)

    def _get_quantized_weight_and_params(self, layer_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
 
        weight = self.quantized_weights.get(layer_name + '_quantized')
        scale = self.quantized_weights.get(layer_name + '_scale')
        zero_point = self.quantized_weights.get(layer_name + '_zero_point')
        return weight, scale, zero_point

    def _dequantize_weights(self, quantized_weight: np.ndarray, scale: np.ndarray, zero_point: np.ndarray) -> np.ndarray:
        #TODO
        return (quantized_weight-zero_point)*scale
    def _calculate_differences(self, weight_1: np.ndarray, weight_2: np.ndarray) -> Optional[Dict[str, float]]:
        #TODO
        if len(weight_1)!=len(weight_2):
            return None
        for i in range(len(weight_1.shape)):
            if len(weight_1[i])!= len(weight_2[i]):
                return None
            else:
                mse = np.mean((weight_1-weight_2)**2)
                max_diff=np.max(np.abs(weight_1-weight_2))
                return {'mse':mse,'max_diff':max_diff}
    def compare(self, layer_name: str) -> Optional[Dict[str, float]]:
        
        weight_1 = self._get_original_weight(layer_name)
        if weight_1 is None:
            return None

        weight_2, scale, zero_point = self._get_quantized_weight_and_params(layer_name)
        if weight_2 is None or scale is None or zero_point is None:
            return None
        weight_2_dequantized = self._dequantize_weights(weight_2, scale, zero_point)

        return self._calculate_differences(weight_1, weight_2_dequantized)


if __name__ == "__main__":
    comparator = ONNXWeightComparator('fasterrcnn_resnet50_fpn.onnx', 'fasterrcnn_resnet50_fpn_quantized.onnx')

    for original_name, original_value in comparator.original_weights.items():
        print(original_name, original_value.shape, original_value.dtype)
        for quantized_name, quantized_value in comparator.quantized_weights.items():
            if original_name in quantized_name:
                print('-', quantized_name, quantized_value.shape, quantized_value.dtype)
        result = comparator.compare(original_name)
        if result is not None:
            print('*', result)