from glob import glob
import os
import onnxruntime as ort
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_onnx_model(model_path: str) -> ort.InferenceSession:
    #TODO
    session = ort.InferenceSession(model_path)
    return session

def preprocess_image(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((224, 224))
    image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)
    return image[np.newaxis, ...] 

def extract_features(session: ort.InferenceSession, image_paths: list[str]) -> np.ndarray:
    features=[]
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for image_path in image_paths:
        img = preprocess_image(image_path).astype(np.float32)
        output=session.run([output_name],{input_name:img})[0]
        features.append(output.flatten())
    return np.array(features)
    #TODO

def cluster_and_analyze(features: np.ndarray, n_clusters: int):
    kmeans=KMeans(n_clusters,random_state=42)
    labels =kmeans.fit_predict(features)
    centers=kmeans.cluster_centers_
    distances=kmeans.transform(features)
    return labels,distances,centers
    #TODO

def main():

    basepath = "/home/project/"
    image_paths = glob(os.path.join(basepath, "images", "*.jpg"))
    n_clusters = 2
    session = load_onnx_model(os.path.join(basepath, "mobilenetv2_features.onnx"))
    features = extract_features(session, image_paths)

    labels, distances, centers = cluster_and_analyze(features, n_clusters)

    print(f"{'Image':<25} {'Cluster':<8} " + "  ".join([f"Dist_to_C{i}" for i in range(n_clusters)]))
    print("=" * (25 + 10 + n_clusters * 15))
    for path, label, dists in zip(image_paths, labels, distances):
        fname = os.path.basename(path)
        dist_str = "        ".join([f"{d:.1f}" for d in dists])
        print(f"{fname:<25} {label:<8} {dist_str}")

if __name__ == '__main__':
    main()