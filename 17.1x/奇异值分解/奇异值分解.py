import numpy as np
from typing import Tuple

def compute_aat(A: np.ndarray) -> np.ndarray:
    A_T= A.transpose(-1,-2)
    AAT=np.matmul(A,A_T)
    return AAT
    # TODO
def eig_decomposition_symmetric(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(M)

    # 只保留实部
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    # 按特征值从大到小排序
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return eigvals, eigvecs
    # TODO

def get_topk_components(eigvals: np.ndarray, eigvecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    topk_vecs = eigvecs[:, :k]
    topk_vals = eigvals[:k]
    return topk_vecs, topk_vals
    # TODO

def compute_singular_values(topk_vals: np.ndarray) -> np.ndarray:
    topk_vals = np.maximum(topk_vals, 0)
    sigma = np.sqrt(topk_vals)
    return sigma
    # TODO

def compute_svd(A: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    AAT = compute_aat(A)
    eigvals, eigvecs = eig_decomposition_symmetric(AAT)
    topk_vecs, topk_vals = get_topk_components(eigvals, eigvecs, k)
    sigma_k = compute_singular_values(topk_vals)
    return topk_vecs, sigma_k

if __name__ == '__main__':

    data = np.arange(-10, 10).reshape(4, 5) / 10
    topk_vecs, sigma_k = compute_svd(data, k=2)
    print(topk_vecs, '\n', sigma_k)