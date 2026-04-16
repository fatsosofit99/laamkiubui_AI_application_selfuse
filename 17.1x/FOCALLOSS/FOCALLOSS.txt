import numpy as np
from typing import Union, Literal

def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-logits))
    #TODO

def softmax(logits: np.ndarray) -> np.ndarray:
    x_exp = np.exp(logits)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / x_sum

    #TODO

def compute_pt(
    probs: np.ndarray,
    labels: np.ndarray,
    task: Literal["binary", "multiclass"]
) -> np.ndarray:
    labels = np.array(labels, dtype=int, copy=True)

    if task == "binary":
        return np.where(labels == 1, probs, 1 - probs)
    elif task == "multiclass":
        rows = np.arange(len(labels))
        return probs[rows, labels]
    else:
        raise ValueError("Unsupported task type.")
    
    #TODO

def focal_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    gamma: float = 2.0,
    alpha: Union[float, np.ndarray] = 0.25,
    task: Literal["binary", "multiclass"] = "binary"
) -> float:    
    if task == "binary":
        probs = sigmoid(logits)
    elif task == "multiclass":
        probs = softmax(logits)
    else:
        raise ValueError("Unsupported task type.")

    pt = compute_pt(probs, labels, task)
    
    if task == "binary":
        # 这题二分类通常按固定 alpha 算，不区分正负类 alpha_t
        loss = -alpha * ((1 - pt) ** gamma) * np.log(pt)
    else:
        if isinstance(alpha, np.ndarray):
            # 报错点修复：确保索引不会越界
            at = alpha[labels]
        else:
            at = alpha
        loss = -at * ((1 - pt) ** gamma) * np.log(pt)

    return float(np.mean(loss))

    #TODO


if __name__ == "__main__":
    
    np.random.seed(42)
    logits_bin = np.array([0.8, -1.2, 2.0])
    labels_bin = np.array([1, 0, 1])

    loss_bin = focal_loss(
        logits=logits_bin,
        labels=labels_bin,
        gamma=2.0,
        alpha=0.25,
        task="binary"
    )
    print("Binary classification Focal Loss:", loss_bin)

    logits_multi = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, 0.3],
        [1.0, 1.0, 1.0]
    ])
    labels_multi = np.array([0, 1, 2])
    alpha_multi = np.array([0.25, 0.25, 0.5])

    loss_multi = focal_loss(
        logits=logits_multi,
        labels=labels_multi,
        gamma=2.0,
        alpha=alpha_multi,
        task="multiclass"
    )
    print("Multiclass classification Focal Loss:", loss_multi)