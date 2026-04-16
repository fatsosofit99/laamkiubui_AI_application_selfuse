from flask import Flask, request, jsonify
import numpy as np
import onnxruntime
import joblib
import threading

app = Flask(__name__)

class ONNXModelSingleton:
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, model_path: str):
        # 使用线程锁保证多线程环境下单例的唯一性
        with cls._lock:
            if model_path not in cls._instances:
                # 只有第一次访问该路径时才创建实例并加载模型
                instance = super().__new__(cls)
                # 加载 ONNX 模型并创建推理会话
                instance.session = onnxruntime.InferenceSession(model_path)
                # 预先获取输入节点名称，方便后续推理
                instance.input_name = instance.session.get_inputs()[0].name
                cls._instances[model_path] = instance
        return cls._instances[model_path]

class SklearnModelSingleton:
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, model_path: str):
        with cls._lock:
            if model_path not in cls._instances:
                instance = super().__new__(cls)
                # 使用 joblib 加载 sklearn 训练好的模型文件（.pkl 或 .joblib）
                instance.model = joblib.load(model_path)
                cls._instances[model_path] = instance
        return cls._instances[model_path]

@app.route("/predict", methods=["POST"])
def predict():
    # 1. 解析请求参数
    req_data = request.get_json()
    model_info = req_data.get("model", {})
    input_list = req_data.get("input", [])
    
    model_type = model_info.get("type")
    model_path = model_info.get("path")
    
    # 将输入数据转换为 numpy 数组，供模型使用
    input_data = np.array(input_list, dtype=np.float32)
    
    # 2. 调用对应的单例类并进行推理
    try:
        if model_type == "onnx":
            # 获取单例（如果已加载则直接返回，否则加载）
            m_instance = ONNXModelSingleton(model_path)
            # ONNX 推理执行
            outputs = m_instance.session.run(None, {m_instance.input_name: input_data})
            prediction = outputs[0]
            
        elif model_type == "sklearn":
            m_instance = SklearnModelSingleton(model_path)
            # Sklearn 推理执行
            prediction = m_instance.model.predict(input_data)
            
        else:
            return jsonify({"error": "Unsupported model type"}), 400
            
        # 3. 返回 JSON 格式结果
        return jsonify({"data": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)