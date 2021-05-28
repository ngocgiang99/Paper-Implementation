from torch.utils.data.dataloader import T
from torchvision import datasets, transforms
from base import BaseDataLoader


from torchvision.transforms import ToTensor
import torch

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class VocDataLoader(BaseDataLoader):
    """
    Pascal VOC dataset loader using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, grid_size=7, nof_box=2):
        self.img_size = (448, 448)
        trsfm = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.ToTensor(),
        ])
        self.grid_size = grid_size
        self.nof_box = nof_box

        target_tranform = self.anotations_tranform
        self.data_dir = data_dir
        self.dataset = datasets.VOCDetection(self.data_dir, image_set="trainval", download=False, transform=trsfm, target_transform=target_tranform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def anotations_tranform(self, x):
        if (type(x) != dict): return x

        labels_map = {  'aeroplane' : 1,
                        'bicycle' : 2,
                        'bird' : 3, 
                        'boat' : 4,
                        'bottle' : 5,
                        'bus' : 6,
                        'car' : 7,
                        'cat' : 8,
                        'chair' : 9,
                        'cow' : 10,
                        'diningtable' : 11,
                        'dog' : 12,
                        'horse' : 13,
                        'motorbike' : 14,
                        'person' : 15,
                        'pottedplant' : 16,
                        'sheep': 17, 
                        'sofa': 18, 
                        'train' : 19,
                        'tvmonitor' : 20}
        
        S = self.grid_size
        B = self.nof_box
        img_size = torch.FloatTensor(self.img_size)

        w = x['annotation']['size']['width']
        h = x['annotation']['size']['height']
        
        w = float(w)
        h = float(h)

        target = torch.zeros((S, S, 5*B + len(labels_map)))
        for obj in x['annotation']['object']:

            xmax = float(obj['bndbox']['xmax']) / w
            xmin = float(obj['bndbox']['xmin']) / w
            ymax = float(obj['bndbox']['ymax']) / h
            ymin = float(obj['bndbox']['ymin']) / h

            xy = torch.FloatTensor([xmin, ymin])
            wh = torch.FloatTensor([xmax-xmin, ymax-ymin])

            xy /= img_size
            wh /= img_size

            uv = (xy * S).ceil()-1
            u,v = int(uv[0]), int(uv[1])

            target[u,v, 4] = 1
            target[u,v, 9] = 1
            target[u,v, int(labels_map[obj['name']]) + 9] = 1

            uv /= S
            delta_xy = xy - uv
            target[u, v, 0:2] = delta_xy
            target[u, v, 2:4] = wh
            target[u, v, 5:7] = delta_xy
            target[u, v, 7:9] = wh
        
        # target = torch.FloatTensor(target)
        # print(target.shape)
        return target




if __name__ == '__main__':
    import os
    import json
    # print(os.getcwd())
    data_loader = VocDataLoader("../data", 4, num_workers=4)
    for batch_idx, (data, target) in enumerate(data_loader):
        # target = VocDataLoader.anotations_tranform(target)
        print(batch_idx, data.shape, target.shape)
        # print(batch_idx, data.shape, json.dumps(target, indent=4, sort_keys=True))
        
        break
