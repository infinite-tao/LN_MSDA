import os
from PIL import Image
import warnings

from torch.utils.data import Dataset
import numpy as np
import bisect
import pandas as pd

from torchvision import transforms
import torch

class ImageList(Dataset):
    def __init__(self, image_root, image_list_root, sample_masks=None, pseudo_labels=None, use_UDS_mask=False):
        self.image_root = image_root
        self.sample_masks = sample_masks
        self.pseudo_labels = pseudo_labels
        self.use_UDS_mask = use_UDS_mask
        self.imgs = image_list_root

        if self.use_UDS_mask:
            self.sample_masks = sample_masks if sample_masks is not None else torch.zeros(len(self.imgs)).float()
            if pseudo_labels is not None:
                self.labels = self.pseudo_labels
                assert len(self.labels) == len(self.imgs), 'Lengths do no match!'
        else:
            if sample_masks is not None:
                temp_list = self.imgs
                self.imgs = [temp_list[i] for i in self.sample_masks]
                if pseudo_labels is not None:
                    self.labels = self.pseudo_labels[self.sample_masks]
                    assert len(self.labels) == len(self.imgs), 'Lengths do no match!'

    def __getitem__(self, index):
        output = {}
        img = np.load(os.path.join(self.image_root, self.imgs[index] + '.npy'))
        class_name = pd.read_excel(
            '/home/ubuntu/zhangyongtao/raw_data_base/label/label.xlsx',
            dtype=object, keep_default_na=False, na_values=[], header=None).values
        for c in class_name:
            global label
            if c[0] == self.imgs[index]:
                label = np.array(float(c[1]))

        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, dim=0)  # shape=(1, 64, 3)
        output['img'] = img
        if self.pseudo_labels is not None:
            output['target'] = torch.LongTensor([np.int64(self.labels[index]).item()])
        else:
            output['target'] = torch.LongTensor([np.int64(label).item()])
        output['idx'] = index
        if self.use_UDS_mask:
            output['mask'] = torch.LongTensor([np.int64(self.sample_masks[index]).item()])

        return output

    def __len__(self):
        return len(self.imgs)
