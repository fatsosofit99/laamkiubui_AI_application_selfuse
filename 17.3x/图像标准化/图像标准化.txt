import os
import cv2
import numpy as np
from typing import List, Tuple

def list_images(image_dir: str) -> List[str]:
    image_paths = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    for file in os.listdir(image_dir):
        if file.lower().endswith(valid_extensions):
            abs_path = os.path.abspath(os.path.join(image_dir, file))
            image_paths.append(abs_path)
    return image_paths
    #TODO

def load_and_preprocess_image(image_path: str) -> np.ndarray:
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32)/255
    return  img_normalized
    #TODO

def compute_image_mean(img: np.ndarray) -> np.ndarray:
    return np.mean(img,axis=(0,1))
    #TODO

def compute_image_std(img: np.ndarray) -> np.ndarray:
    return np.std(img,axis=(0,1))
    #TODO

def compute_global_mean_std(mean_list: List[np.ndarray], std_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    global_mean = np.mean(mean_list, axis=0)
    global_std = np.mean(std_list, axis=0)
    return global_mean, global_std
    #TODO

def normalize_image(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (img-mean)/std
    #TODO

if __name__ == '__main__':
    image_dir = '/home/project/images'

    image_paths = list_images(image_dir)
    print(f"找到 {len(image_paths)} 张图像")

    mean_list = []
    std_list = []
    for path in image_paths:
        img = load_and_preprocess_image(path)
        mean = compute_image_mean(img)
        std = compute_image_std(img)
        mean_list.append(mean)
        std_list.append(std)

    global_mean, global_std = compute_global_mean_std(mean_list, std_list)
    print(f"\n全局均值: {global_mean}")
    print(f"全局标准差: {global_std}")

    sample_img = load_and_preprocess_image(image_paths[0])
    norm_img = normalize_image(sample_img, global_mean, global_std)