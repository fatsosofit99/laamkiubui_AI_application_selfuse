import json
from typing import List

def dice_loss(pred: List, labels: List, smooth: float=1e-6)->float:
    #TODO
    intersection = 0.0
    p_sum = 0.0
    l_sum = 0.0

    for n in range(len(pred)):
        for c in range(len(pred[n])):
            for h in range(len(pred[n][c])):
                for w in range(len(pred[n][c][h])):
                    p = float(pred[n][c][h][w])
                    l = float(labels[n][c][h][w])

                    intersection += p * l
                    p_sum += p
                    l_sum += l

    dice = (2 * intersection + smooth) / (p_sum + l_sum + smooth)
    return 1 - dice

if __name__ == '__main__':
    json_path = '/home/project/data.json'
    json_dict = {}
    with open(json_path, 'r') as json_file:
        json_dict = json.load(json_file)
    pred = json_dict['pred_mask']
    labels = json_dict['labels_mask']
    loss = dice_loss(pred, labels)
    print(loss)