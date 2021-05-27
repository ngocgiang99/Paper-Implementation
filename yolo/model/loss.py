import torch.nn.functional as F
import torch

from utils.util import compute_iou

def nll_loss(output, target):
    return F.nll_loss(output, target)

def yolov1_loss(output, target, hyper_param):
    S = hyper_param['S']
    B = hyper_param['B']
    C = hyper_param['C']

    lbd_coord = hyper_param['lbd_coord']
    lbd_noobj = hyper_param['lbd_noobj']

    one_obj = torch.zeros((output.shape[0], S, S, B))


    for it, o in enumerate(output): 
        S = output.shape[0]
        B = 2
        C = 20

        for i in range(S*S):
            u = i // S
            v = i - u * S
            max_iou = 0.0
            for j in range(B):
                one_obj[it, u, v, j] = None
                for k, t in enumerate(target[it]):
                    iou = compute_iou(output[it, u, v, j*5+1: (j+1)*5], t)
                    if iou > max_iou:
                        max_iou = iou
                        one_obj[it, u, v, j] = torch.FloatTensor([t[0], t[2], t[1] - t[0], t[3] - t[2], t[4]])

    _target_onehot = torch.FloatTensor(output.shape[0], S, S, B, C)
    _target_onehot.zero_()
    _target_onehot.scatter_(1, one_obj[:, :, :, :, 4], 1)

    loss_xy = 0.0 # Loss for x,y coordinate
    loss_wh = 0.0 # Loss for width and height of image
    loss_conf = 0.0 # Loss for confidence prediction
    for i in range(B):
        
        j = i*5
        loss_xy += (output[:, :, :, [j+1,j+2]] - one_obj[:, :, :, i, [0,1]]) ** 2
        loss_wh += (torch.sqrt(output[:, :, :, [j+3, j+4]]) - torch.sqrt(one_obj[:, :, :, i, [2,3]]) ) ** 2
        loss_conf += (output[:, :, :, 2*B:] - _target_onehot[:, :, :, i, :])

    

    
