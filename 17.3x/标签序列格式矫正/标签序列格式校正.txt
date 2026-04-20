from typing import List, Tuple

def fix_bio_labels(labels: List[str]) -> Tuple[List[str], bool]:
    #TODO
    fixed_labels =labels[:]
    changed=False
    for i in range(len(fixed_labels)):
        if fixed_labels[i]=='I':
            if i==0 or fixed_labels[i-1] not in ('B','I'):
                fixed_labels[i]='B'
                changed = True
    return fixed_labels,changed
        
if __name__ == "__main__":
    test_cases = [
        ["O", "B", "I", "I", "O", "B", "I"],
        ["I", "I", "O"],
        ["O", "I", "O"],
        ["B", "I", "I"],
        ["O", "I", "I", "O", "I", "O", "I"],
        ["B", "B", "B"],
        ["O", "O", "O"],
        ["O", "B", "I", "O", "B", "O", "B", "I", "I"]
    ]

    for seq in test_cases:
        
        fixed, changed = fix_bio_labels(seq)
        if changed:
            print(f'序列 {seq} 不合法，修改后：{fixed}')
        else:
            print(f'序列 {seq} 合法')