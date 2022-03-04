import os
import torch
import numpy as np
from scipy.io import loadmat
from torchvision import datasets


class Dataset(torch.utils.data.Dataset):
    '''
        Dataset for face images and shape images.
        face images contain a series of face and corresponding parameters
    '''

    def __init__(self, image_path, shape_path=None, loc_idx=None, transform=None):
        self.image_path = image_path
        self.shape_path = shape_path
        # datasets
        self.image_dataset = datasets.ImageFolder(image_path)
        self.shape_dataset = None
        if shape_path is not None:
            self.shape_dataset = datasets.ImageFolder(shape_path)
        self.transform = transform
        self.loc_idx = loc_idx

    def __getitem__(self, index):
        if self.shape_dataset is None:
            image, _ = self.image_dataset.imgs[index]
        else:
            shape, _ = self.shape_dataset.imgs[index]
            image_len = len(self.image_dataset.imgs)
            image, _ = self.image_dataset.imgs[index % image_len]

        filename = os.path.splitext(image)[0]
        config_file = filename + '.mat'
        config = loadmat(config_file)

        face_shape = config['face']
        tri = config['tri']
        gamma = config['gamma']
        intrinsic = config['intrinsic']
        locs = config['locs_{}'.format(self.loc_idx)]
        uv_coords = config['uv'].astype(np.float32)

        if self.transform:
            image = self.transform(image)
            if self.shape_dataset:
                shape = self.transform(shape)

        if self.shape_dataset:
            return image, shape, face_shape, tri, gamma, intrinsic, uv_coords, locs
        return image, face_shape, tri, gamma, intrinsic, uv_coords, locs

    def __len__(self):
        _len = len(self.image_dataset.imgs)
        if self.shape_dataset:
            _len = len(self.shape_dataset.imgs)
        return _len
