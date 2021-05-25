import torch.nn.functional as F
import torch

from utils.util import compute_iou

def nll_loss(output, target):
    return F.nll_loss(output, target)

def yolov1_loss(output, target):
    one_obj = torch.zeros(output.shape)

    for it, o in enumerate(output): 
        S = output.shape[0]
        B = 2
        C = 20

        for i in range(S*S):
            u = i // S
            v = i - u * S
            max_iou = 0.0
            for j in range(B):
                
                for k, t in enumerate(target[it]):
                    iou = compute_iou(output[it, u, v, j, j*5: (j+1)*5], t)
                    if iou > max_iou:
                        one_