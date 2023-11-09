import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))

from PIL import Image

import groundingdino.datasets.transforms as T
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch



image_path = {}
image_label = {}


class BirdDataset(Dataset):
    """
    # Description:
        Dataset for retrieving CUB-200-2011 images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', resize=(448,448), index=False, dataset_path=''):
        assert phase in ['train', 'val', 'test','traintest']
        self.phase = phase
        self.resize = resize
        self.image_id = []
        self.obj_loc = []
        self.num_classes = 200
        self.DATAPATH = os.path.join(dataset_path,'CUB_200_2011/')
        

        # get image path from images.txt
        with open(os.path.join(self.DATAPATH, 'images.txt')) as f:
            for line in f.readlines():
                id, path = line.strip().split(' ')
                image_path[id] = path

        # get image label from image_class_labels.txt
        with open(os.path.join(self.DATAPATH, 'image_class_labels.txt')) as f:
            for line in f.readlines():
                id, label = line.strip().split(' ')
                image_label[id] = int(label)


        # get train/test image id from train_test_split.txt
        with open(os.path.join(self.DATAPATH, 'train_test_split.txt')) as f:
            for line in f.readlines():
                image_id, is_training_image = line.strip().split(' ')
                is_training_image = int(is_training_image)

                if self.phase == 'train' and is_training_image:
                    self.image_id.append(image_id)
                if self.phase in ('val', 'test') and not is_training_image:
                    self.image_id.append(image_id)
        self.transform =  T.Compose(
                            [
                                T.RandomResize([800], max_size=1333),
                                T.ToTensor(),
                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            ]
                        )
        self.index = index

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]
        image_full_path = os.path.join(self.DATAPATH, 'images', image_path[image_id])

        # load image
        image_pil = Image.open(image_full_path).convert("RGB")  # load image

        transform = transforms.Compose(
            [
                transforms.Resize([448,448]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image = transform(image_pil)  # 3, h, w

        return image, image_label[image_id] - 1,

    def __len__(self):
        return len(self.image_id)


if __name__ == '__main__':
    ds = BirdDataset(phase='val',dataset_path='/root/autodl-tmp')

    print(len(ds))
    for i in range(100):
        image_pil, image, label  = ds[i]
        print(image.shape)
        print(label)
