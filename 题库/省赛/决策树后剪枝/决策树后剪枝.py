import copy
from collections import Counter
from typing import Dict, List, Union, Optional, Tuple


Tree = Dict[str, Union[str, Dict[str, 'Tree']]]
Sample = Dict[str, str]
ValidationData = List[Sample]
def predict(decision_tree: Tree, sample: Sample) -> str:
    #TODO
    if is_leaf(decision_tree):
        return decision_tree

    keys = next(iter(decision_tree))
    value = sample[keys]
    next_node = decision_tree[keys][value]
    return predict(next_node,sample)

def evaluate_accuracy(decision_tree: Tree, validation_data: ValidationData) -> float:
    correct = 0
    for sample in validation_data:
        pred_label = predict(decision_tree, sample)
        if pred_label == sample['label']:
            correct += 1
    return correct / len(validation_data)

def filter_data_for_subtree(validation_data: ValidationData, decision_path: List[Tuple[str, str]]) -> ValidationData:
    #TODO
    # print(decision_path)

    validation_sample = []
    for sample in validation_data:
        is_value = True
        for key,value in decision_path:
            if sample[key] != value:
                is_value = False
                break
        if is_value:
            validation_sample.append(sample)
    # print(validation_sample)
    return validation_sample
        

def majority_label(node_data: ValidationData) -> str:
    #TODO
    T = F = 0
    for sample in node_data:
        if sample['label'] == '好瓜':
            T += 1
        else:
            F += 1
    if T >= F:
        return '好瓜'
    else:
        return '坏瓜'
        
def is_leaf(node: Union[str, Tree]) -> bool:
    #TODO
    if type(node) is str:
        return True
    else:
        return False

def post_prune(decision_tree: Tree,
               validation_data: ValidationData,
               decision_path: Optional[List[Tuple[str, str]]] = None) -> Tree:

    if decision_path is None:
        decision_path = []

    if is_leaf(decision_tree):
        return decision_tree

    attribute = list(decision_tree.keys())[0]
    subtrees = decision_tree[attribute]

    for branch_value in list(subtrees.keys()):
        child = subtrees[branch_value]
        if not is_leaf(child):
            child_path = decision_path + [(attribute, branch_value)]
            pruned_child = post_prune(child, validation_data, decision_path=child_path)
            subtrees[branch_value] = pruned_child

    node_data = filter_data_for_subtree(validation_data, decision_path)

    if not node_data:
        return decision_tree

    leaf_label = majority_label(node_data)

    original_accuracy = evaluate_accuracy(decision_tree, node_data)

    counts = Counter([s['label'] for s in node_data])
    pruned_accuracy = counts[leaf_label] / len(node_data)

    if pruned_accuracy > original_accuracy:
        return leaf_label
    else:
        return decision_tree


def main() -> None:
    decision_tree = {
        '脐部': {
            '凹陷': {
                '色泽': {
                    '青绿': '好瓜',
                    '乌黑': '好瓜',
                    '浅白': '坏瓜',
                }},
            '稍凹': {
                '根蒂': {
                    '稍蜷': {
                        '色泽': {
                            '乌黑': 
                                {'纹理': {
                                    '稍糊': '好瓜',
                                    '清晰': '坏瓜',
                                    '模糊': '好瓜'
                                }},
                            '青绿': '好瓜',
                            '浅白': '好瓜'
                            },},
                    '蜷缩': '坏瓜',
                    '硬挺': '好瓜' 
                    },},
             '平坦': '坏瓜'   
            }}

    validation_data = [
        {'脐部': '平坦', '色泽': '乌黑', '纹理': '模糊', '根蒂': '稍蜷', 'label': '坏瓜'}, 
        {'脐部': '凹陷', '色泽': '浅白', '纹理': '模糊', '根蒂': '硬挺', 'label': '坏瓜'}, 
        {'脐部': '凹陷', '色泽': '青绿', '纹理': '清晰', '根蒂': '硬挺', 'label': '好瓜'}, 
        {'脐部': '凹陷', '色泽': '乌黑', '纹理': '模糊', '根蒂': '稍蜷', 'label': '坏瓜'}, 
        {'脐部': '稍凹', '色泽': '浅白', '纹理': '模糊', '根蒂': '硬挺', 'label': '坏瓜'}, 
        {'脐部': '平坦', '色泽': '浅白', '纹理': '稍糊', '根蒂': '稍蜷', 'label': '坏瓜'},
        {'脐部': '平坦', '色泽': '浅白', '纹理': '清晰', '根蒂': '硬挺', 'label': '坏瓜'},
        {'脐部': '平坦', '色泽': '乌黑', '纹理': '清晰', '根蒂': '蜷缩', 'label': '坏瓜'},
        {'脐部': '平坦', '色泽': '乌黑', '纹理': '模糊', '根蒂': '蜷缩', 'label': '好瓜'},
        {'脐部': '平坦', '色泽': '乌黑', '纹理': '清晰', '根蒂': '稍蜷', 'label': '坏瓜'},
        {'脐部': '凹陷', '色泽': '浅白', '纹理': '模糊', '根蒂': '蜷缩', 'label': '好瓜'},
        {'脐部': '稍凹', '色泽': '浅白', '纹理': '模糊', '根蒂': '蜷缩', 'label': '坏瓜'},
        {'脐部': '稍凹', '色泽': '乌黑', '纹理': '稍糊', '根蒂': '稍蜷', 'label': '坏瓜'},
        {'脐部': '凹陷', '色泽': '浅白', '纹理': '清晰', '根蒂': '稍蜷', 'label': '好瓜'},
        {'脐部': '稍凹', '色泽': '乌黑', '纹理': '清晰', '根蒂': '蜷缩', 'label': '坏瓜'},
        {'脐部': '凹陷', '色泽': '青绿', '纹理': '稍糊', '根蒂': '蜷缩', 'label': '坏瓜'},
        {'脐部': '平坦', '色泽': '乌黑', '纹理': '模糊', '根蒂': '蜷缩', 'label': '坏瓜'},
        {'脐部': '稍凹', '色泽': '青绿', '纹理': '清晰', '根蒂': '硬挺', 'label': '好瓜'},
        {'脐部': '平坦', '色泽': '乌黑', '纹理': '模糊', '根蒂': '硬挺', 'label': '坏瓜'},
        {'脐部': '凹陷', '色泽': '青绿', '纹理': '稍糊', '根蒂': '硬挺', 'label': '好瓜'}]

    print("==== 剪枝前准确率 ====")
    acc_before = evaluate_accuracy(decision_tree, validation_data)
    print(f"Accuracy = {acc_before*100:.2f}%")

    pruned_tree = post_prune(decision_tree, validation_data)

    print("\n==== 剪枝后决策树结构 ====")
    print(pruned_tree)

    print("\n==== 剪枝后准确率 ====")
    acc_after = evaluate_accuracy(pruned_tree, validation_data)
    print(f"Accuracy = {acc_after*100:.2f}%")


if __name__ == "__main__":
    main()