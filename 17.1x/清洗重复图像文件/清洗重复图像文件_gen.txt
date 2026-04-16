import cv2
import os
from typing import List, Dict, Tuple

def group_duplicate_images(folder: str) -> List[List[str]]:
    groups = {}
    # 1. 递归遍历 (os.walk 是最标准的做法)
    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            img = cv2.imread(path)
            
            if img is not None:
                # 2. 将像素矩阵转为字节作为字典的键 (Key)
                # 只要像素完全一致，tobytes() 的结果就完全一致
                content = img.tobytes()
                
                # 3. 简单的字典分组逻辑
                if content not in groups:
                    groups[content] = []
                groups[content].append(path)
    # 只返回路径数量大于 1 的组
    return [p for p in groups.values() if len(p) > 1]
    # TODO


if __name__ == '__main__':
    groups = group_duplicate_images('/home/project/dataset')
    for g in groups:
        print(g)