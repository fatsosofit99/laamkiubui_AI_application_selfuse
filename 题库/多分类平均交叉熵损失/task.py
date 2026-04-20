from typing import List
import json

def cross_entropy_loss(logits: List, labels: List)->float:
    """稳定版
    num=len(logits)
    loss=0
    for i in range(num):
        logit,idx=logits[i],labels[i]
        max_val = max(logit)
        exp_val=[math.exp(x-max_val) for x in logit]
        sum_exp=sum(exp_val)

        prob_true = exp_val[idx]/sum_exp
        sample_loss=-math.log(prob_true)
        loss+=sample_loss
    return loss/num
    
    """ 
    #TODO
    import math
    total_loss = 0.0
    n = len(logits)
    
    for i in range(n):
        sample_logits = logits[i]
        target_class = labels[i]
        
        # --- 第一步：计算 Softmax 概率 (原始定义) ---
        # 计算所有类别的 exp(x)
        exp_values = [math.exp(x) for x in sample_logits]
        # 计算分母 sum(exp(x))
        sum_exp = sum(exp_values)
        # 计算每个类别的概率 P
        probabilities = [ev / sum_exp for ev in exp_values]
        
        # --- 第二步：计算交叉熵 ---
        # 按照公式 -sum(y_j * log(p_j))
        # 只有真实标签 y_j = 1，其余为 0，所以只取 target_class 对应的概率
        p_target = probabilities[target_class]
        
        # 这里使用 math.log (自然对数 ln)
        # 如果要对齐之前图片的 log2，请改为 math.log2
        sample_loss = -math.log(p_target)
        
        total_loss += sample_loss
        
    # 返回平均值
    return total_loss / n
if __name__ == '__main__':
    json_dict = {}
    with open('/home/project/data.json', 'r') as json_file:
        json_dict = json.load(json_file)
    labels, logits = json_dict['labels'], json_dict['logits']
    loss = cross_entropy_loss(logits, labels) 
    print(loss) # 2.560082197189331