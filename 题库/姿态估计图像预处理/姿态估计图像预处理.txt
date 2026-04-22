
import numpy as np
from PIL import Image
from typing import Tuple
from matplotlib import pyplot as plt
import scipy.io
from torch.utils.data import Dataset
class PoseDataset(Dataset):
    def __init__(self, img_dir: str, anno_file: str, padding: int = 5) -> None:
        self.img_dir = img_dir
        self.anno = scipy.io.loadmat(anno_file)['joints']
        self.padding = padding

    def __len__(self) -> int:
        #TODO
        return self.anno.shape[2]

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray, np.ndarray]:
        #TODO
        img_path = f"{self.img_dir}/im{idx+1:04d}.jpg"
        img = Image.open(img_path)
        img = np.array(img)
        keypoints = np.array(self.anno[:2,:,idx])
        img,keypoint_loc = self.get_crop(img,keypoints)

        return img_path,img,keypoint_loc

    def get_crop(self, image: np.ndarray, keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        #TODO
        # image:HWC
        H,W = image.shape[:2]
        x_min,y_min = np.min(keypoints,axis=1).astype(np.int32)
        x_max,y_max = np.max(keypoints,axis=1).astype(np.int32)

        # key_x ,key_y= keypoints[0],keypoints[1]
        # # ✅ 图像像素坐标只能是整数（image[y_min:y_max, x_min:x_max] 切片必须用整数）
        # x_min = int(key_x.min())
        # x_max = int(key_x.max())
        # y_min = int(key_y.min())
        # y_max = int(key_y.max())

        x_min -= self.padding
        x_max += self.padding
        y_min -= self.padding
        y_max += self.padding

        x_min = max(0,x_min)
        x_max = min(W,x_max)
        y_min = max(0,y_min)
        y_max = min(H,y_max)

        cropped_image = image[y_min:y_max,x_min:x_max]

        Δ = np.array([x_min,y_min]).reshape(2,1)
        new_key = keypoints-Δ
        # new_key =keypoints.copy()
        # new_key[0] = key_x-x_min
        # new_key[1] = key_y-y_min
        return cropped_image,new_key