from typing import List, Tuple
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def read_ip_data_from_file(filename: str) -> Tuple[List[str], np.ndarray]:
    #TODO
    ips=[]
    labels=[]
    with open(filename,'r')as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.splite(',')if ',' in line else line.split()
            if len(parts)>=2:
                ips.append(parts[0].strip())
                labels.append(parts[1].strip())
    return ips,np.array(labels,dtype=int)
def convert_ip_to_vector(ip_addresses: List[str]) -> np.ndarray:
    #TODO
    vectors=[]
    for ip in ip_addresses:
        parts =[int(x) for x in ip.split('.')]
        vectors.append(parts)
    return np.array(vectors)
    


def tsne_reduce(ip_vectors: np.ndarray, n_components: int = 2) -> np.ndarray:
    #TODO
    tsne=TSNE(n_components=n_components,random_state=42)
    return tsne.fit_transform(ip_vectors)


def visualize_2d(ip_vectors_2d: np.ndarray, labels: np.ndarray, title: str, xlabel: str, ylabel: str) -> plt.Figure:
    #TODO
    # fig = plt.figure(figsize=(8, 6))
    
    fig=plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    scatter=ax.scatter(ip_vectors_2d[:,0],ip_vectors_2d[:,1],c=labels,cmap='viridis',alpha=0.7)
    # 设置标题和坐标轴标签
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # 添加颜色条辅助说明
    fig.colorbar(scatter, ax=ax, label='Label')
    return fig


def main() -> None:
    
    ip_addresses, labels = read_ip_data_from_file('ip_data.txt')
    ip_vectors = convert_ip_to_vector(ip_addresses)
    
    ip_vectors_2d = tsne_reduce(ip_vectors)
    fig = visualize_2d(ip_vectors_2d, labels, 't-SNE Visualization of IP Addresses', 't-SNE 1', 't-SNE 2')
    fig.savefig("tsne_visualization.png")


if __name__ == '__main__':
    main()