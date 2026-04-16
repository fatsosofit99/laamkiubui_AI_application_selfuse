from sklearn.model_selection import train_test_split
from typing import List
import numpy as np
import pandas as pd
import joblib
from sklearn import metrics


def load_models(model_paths: List[str]) -> List:
    #TODO
    models = []
    for path in model_paths:
        model=joblib.load(path)
        models.append(model)
    return models

def predict_all(models: List, X: np.ndarray) -> np.ndarray:
    #TODO
    result=[]
    for model in models:
        pred = model.predict(X)
        result.append(pred)
    return np.array(result)


def weighted_average(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result =np.dot(weights,predictions)
    return result
    #TODO
"""
更通用方法
def weighted_average(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # axis=0 表示对“模型数量”这一层进行加权
    return np.average(predictions, axis=0, weights=weights)
"""

def evaluate_mse(y_true: np.ndarray, predictions: np.ndarray) -> float:
    sum,count=0,0
    for i in range(len(y_true)):
        sum+=(y_true[i]-predictions[i])**2
        count+=1
    return float(sum/count)
    #TODO


if __name__ == "__main__":
    
    raw_df = pd.read_csv('boston.txt', sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    model_paths = ["/home/project/model_rf.joblib", "/home/project/model_gb.joblib"]
    models = load_models(model_paths)

    preds = predict_all(models, X_test)
    weights = [0.1, 0.9]
    fused_preds = weighted_average(preds, weights)

    individual_mses = [evaluate_mse(y_test, pred) for pred in preds]
    fusion_mse = evaluate_mse(y_test, fused_preds)

    print("各模型 MSE：")
    for i, mse in enumerate(individual_mses):
        print(f"Model {i + 1}: {mse:.4f}")
    print(f"\n融合模型 MSE: {fusion_mse:.4f}")