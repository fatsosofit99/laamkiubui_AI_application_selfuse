from typing import List, Tuple
import numpy as np


def k_fold_cross_validation(X: List, k: int=5, random_seed: int=11, shuffle: bool=False)->Tuple[List, List]:
    n = len(X)
    indices = list(range(n))
    
    # 如果设置了 shuffle，则根据随机种子打乱索引
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
    train_indices_list = []
    test_indices_list = []
    
    # 计算每一折的基础大小以及无法整除时的余数
    base_fold_size = n // k
    remainder = n % k
    
    start_idx = 0
    for i in range(k):
        # 前 remainder 个折的样本数比基础大小多 1
        current_fold_size = base_fold_size + 1 if i < remainder else base_fold_size
        end_idx = start_idx + current_fold_size
        
        # 划分验证集和训练集索引
        test_idx = indices[start_idx:end_idx]
        train_idx = indices[:start_idx] + indices[end_idx:]
        
        test_indices_list.append(test_idx)
        train_indices_list.append(train_idx)
        
        # 更新下一折的起始位置
        start_idx = end_idx
        
    return train_indices_list, test_indices_list
    #TODO
    

if __name__ == '__main__':
    # 示例数据集：包含 5 张分辨率为 (3, 224, 224) 图像
    data = np.random.rand(5, 3, 224, 224).tolist()
    
    # 示例一：划分数据集前不打乱原数据的顺序
    k = 5
    train_indices_list, test_indices_list = k_fold_cross_validation(data, k)
    
    # 打印结果
    for i in range(len(train_indices_list)):
        print(f'Fold {i+1}: {train_indices_list[i]}, {test_indices_list[i]}')
   
    '''
        Fold 1: [1, 2, 3, 4], [0]
        Fold 2: [0, 2, 3, 4], [1]
        Fold 3: [0, 1, 3, 4], [2]
        Fold 4: [0, 1, 2, 4], [3]
        Fold 5: [0, 1, 2, 3], [4]
    '''