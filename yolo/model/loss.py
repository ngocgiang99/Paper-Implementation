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

    one_obj_mask = target[:, :, :, 4] > 0
    no_obj_mask = target[:, :, :, 4] == 0
    one_obj_mask = one_obj_mask.unsqueeze(-1).expand_as(target)
    no_obj_mask = no_obj_mask.unsqueeze(-1).expand_as(target)

    # No object loss
    output_no_obj = output[no_obj_mask].view(-1, 30)
    target_no_obj = target[no_obj_mask].view(-1, 30)
    no_obj_conf_idx = [4,9]
    no_obj_loss = F.mse_loss(output_no_obj[:, no_obj_conf_idx], target_no_obj[:, no_obj_conf_idx], reduction='sum')

    # Coordinate loss and width-height loss
    




def test_yolov1_loss():
    from data_loader.data_loaders import VocDataLoader

    data_loader = VocDataLoader("../data", 4, num_workers=4)
    test_target = None
    for batch_idx, (data, target) in enumerate(data_loader):
        # target = VocDataLoader.anotations_tranform(target)
        print(batch_idx, data.shape, target.shape)
        # print(batch_idx, data.shape, json.dumps(target, indent=4, sort_keys=True))
        test_target = target.clone()
        break

    test_output = torch.rand((4, 7, 7, 30))
    hyper_param = {
        'S': 7,
        'B': 2,
        'C': 20,
        'lbd_coord': 5,
        'lbd_noobj': 0.5
    }
    loss = yolov1_loss(test_output, test_target, hyper_param)
    print(loss)

if __name__ == '__main__':

    test_yolov1_loss()
    




    
