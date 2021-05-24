from torchvision import datasets, transforms
from base import BaseDataLoader


from torchvision.transforms import ToTensor

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

        # target_tranform = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        self.data_dir = data_dir
        self.dataset = datasets.VOCDetection(self.data_dir, image_set="trainval", download=False, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == '__main__':
    import os
    import json
    # print(os.getcwd())
    data_loader = VocDataLoader("../data", 1)
    for batch_idx, (data, target) in enumerate(data_loader):
        # print(batch_idx, data.shape, target.shape)
        print(batch_idx, data.shape, json.dumps(target, indent=4, sort_keys=True))
        break
