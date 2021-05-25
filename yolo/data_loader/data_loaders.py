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
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
            # transforms.ToTensor(),
        ])

        target_tranform = VocDataLoader.anotations_tranform
        self.data_dir = data_dir
        self.dataset = datasets.VOCDetection(self.data_dir, image_set="trainval", download=False, transform=trsfm, target_transform=target_tranform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def anotations_tranform(x):
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
        
        try:
            w = x['annotation']['size']['width']
            h = x['annotation']['size']['height']
            
            w = float(w)
            h = float(h)

            cnt = 0
            anno = []
            for obj in x['annotation']['object']:
                xmax = float(obj['bndbox']['xmax']) / w
                xmin = float(obj['bndbox']['xmin']) / w
                ymax = float(obj['bndbox']['ymax']) / h
                ymin = float(obj['bndbox']['ymin']) / h

                anno.append([labels_map[obj['name']], xmin, xmax, ymin, ymax])
                cnt += 1
            
            while cnt < 30:
                anno.append([-1, -1, -1, -1, -1])
                cnt += 1
            
            anno = torch.FloatTensor(anno)
            print(anno.shape)
            return anno
        except:
            print(x)







if __name__ == '__main__':
    import os
    import json
    # print(os.getcwd())
    data_loader = VocDataLoader("../data", 4, num_workers=4)
    for batch_idx, (data, target) in enumerate(data_loader):
        target = VocDataLoader.anotations_tranform(target)
        print(batch_idx, data.shape, target.shape)
        # print(batch_idx, data.shape, json.dumps(target, indent=4, sort_keys=True))
        
        break
