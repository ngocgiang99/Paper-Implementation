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

    

    
