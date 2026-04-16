import jieba.posseg as pseg
import numpy as np
from typing import List, Tuple, Dict

def tokenize_with_pos(sentences: List[str]) -> List[List[Tuple[str, str]]]:
    result=[]
    for sentence in sentences:
        words = pseg.cut(sentence)
        sent_tokens=[]
        for word in words:
            sent_tokens.append((word.word, word.flag))
        result.append(sent_tokens)
    return result
    #TODO

def build_pos_vocab(data: List[List[Tuple[str, str]]]) -> Dict[str, int]:
    tag_set = set()
    for sent in data:
        for _,tag in sent:
            tag_set.add(tag)
    sorted_tags = sorted(tag_set)
    tag_to_idx = {tag: idx for idx, tag in enumerate(sorted_tags)}
    return tag_to_idx

    #TODO

def encode_pos_onehot(data: List[List[Tuple[str, str]]], tag_to_idx: Dict[str, int]) -> List[np.ndarray]:
    results = []
    num_tags = len(tag_to_idx)

    for sent in data:
        onehot_matrix = []
        for _, tag in sent:
            vec = np.zeros(num_tags, dtype=int)
            if tag in tag_to_idx:
                vec[tag_to_idx[tag]] = 1
            onehot_matrix.append(vec)
        results.append(np.array(onehot_matrix))
    
    return results

    #TODO


if __name__ == '__main__':
    sentences = ["我爱北京天安门", "今天是个好天气"]
        
    tokenized_data = tokenize_with_pos(sentences)
    print("词性标注结果:")
    for i, sent in enumerate(tokenized_data):
        print(f"句子 {i+1}:", sent)

    tag_to_idx = build_pos_vocab(tokenized_data)
    print("\n词性词表（tag_to_idx）:", tag_to_idx)

    onehot_results = encode_pos_onehot(tokenized_data, tag_to_idx)
    print("\nOne-hot 编码矩阵:")
    for i, mat in enumerate(onehot_results):
        print(f"句子 {i+1}:\n{mat}")