import os
import torch
import random
import numpy as np
from mtcnn.mtcnn import MTCNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def collate_pil(x):
    out_x, out_y = [], []
    for xx, yy in x:
        out_x.append(xx)
        out_y.append(yy)
    return out_x, out_y


if __name__ == '__main__':

    torch.seed(117)
    random.seed(117)
    np.random.seed(117)
    torch.manual_seed(117)

    batch_size = 1
    workers = 0 if os.name == 'nt' else 8
    dataset_dir = r'..\Auxiliary\FaceBank'
    cropped_dataset = r'..\Auxiliary\ClippedFaceBank'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(
        image_size=(300, 300), margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    dataset = datasets.ImageFolder(
        dataset_dir, transform=transforms.Resize((512, 512)))
    dataset.samples = [
        (p, p.replace(dataset_dir, cropped_dataset))
        for p, _ in dataset.samples
    ]
    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=collate_pil
    )

    for i, (x, y) in enumerate(loader):
        x = mtcnn(x, save_path=y, save_landmarks=True)
