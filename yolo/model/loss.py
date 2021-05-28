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

    N = output.shape[0]

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
    # print(output[one_obj_mask].shape)
    # print(target[one_obj_mask].shape)
    output_obj = output[one_obj_mask].view(-1, 30)  # Get cell contains box from yolo output
    target_obj = target[one_obj_mask].view(-1, 30)  # Get cell contains box from VOC annotation

    pred_box = output_obj[:, :10].contiguous().view(-1, 5)          # Get [x,y,w,h,c] from output
    target_box = target_obj[:, :10].contiguous().view(-1, 5)        #

    # print(target_box.shape)
    target_obj_iou = torch.FloatTensor(target_box.shape)

    coo_response_obj = torch.zeros(target_box.shape, dtype=torch.bool)
    # print(coo_response_obj.shape)

    for i in range(0, pred_box.shape[0], 2):
        o_box = pred_box[i:i+2]
        # print("obox shape: ", o_box.shape)
        o_box_xywh = torch.zeros(o_box.shape)
        o_box_xywh[:, 2:4] = o_box[:, 2:4]
        o_box_xywh[:, 0:2] = o_box[:, 0:2] - 0.5 * o_box[:, 2:4]

        t_box = target_box[i].view(-1, 5)
        # print("t box shape: ", t_box.shape)
        t_box_xywh = torch.zeros(t_box.shape)
        t_box_xywh[:, 2:4] = t_box[:, 2:4]
        t_box_xywh[:, 0:2] = t_box[:, 0:2] - 0.5 * t_box[:, 2:4]
        iou = compute_iou(o_box_xywh, t_box_xywh)
        # print(iou)
        max_iou, max_index = iou.max(0)
        # print(i, max_iou, max_index)

        max_index = max_index.data
        coo_response_obj[i+max_index] = True
        target_obj_iou[i+max_index, 4] = max_iou.data

    target_obj_iou = target_obj_iou.cuda()
    # print(coo_response_obj)

    # print(pred_box[coo_response_obj].shape)
    output_pred = pred_box[coo_response_obj].view(-1, 5).cuda()
    target_pred = target_box[coo_response_obj].view(-1, 5).cuda()
    target_iou_pred = target_obj_iou[coo_response_obj].view(-1,5).cuda()

    # print(output_pred.shape, target_pred.shape)
    xy_loss = F.mse_loss(output_pred[:, 0:2], target_pred[:, 0:2], reduction='sum')
    wh_loss = F.mse_loss(torch.sqrt(output_pred[:, 2:4]), torch.sqrt(target_pred[:, 2:4]), reduction='sum')
    obj_conf_loss = F.mse_loss(output_pred[:, 4], target_iou_pred[:, 4], reduction='sum')
    loc_loss = xy_loss + wh_loss

    # Classes Loss
    output_class = output[one_obj_mask][10:].cuda()
    target_class = target[one_obj_mask][10:].cuda()
    classes_loss = F.mse_loss(output_class, target_class, reduction='sum')



    # Sum loss
    loss = lbd_coord * loc_loss + obj_conf_loss + lbd_noobj * no_obj_loss + classes_loss
    return loss / N


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
    




    
