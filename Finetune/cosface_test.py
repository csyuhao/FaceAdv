import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from module.units.cosface_module import CosFace


def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()


if __name__ == "__main__":

    random.seed(117)
    np.random.seed(117)
    torch.manual_seed(117)
    torch.cuda.manual_seed(117)

    transform = transforms.Compose([
                    transforms.Resize((112, 96)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cosface = CosFace(classnum=156, pretrained=None).to(device)

    dataset_dir = r'..\Auxiliary\TestFaceBank\lfw-'
    dataset = datasets.ImageFolder(
        dataset_dir, transform=transform)

    batch_size = 32
    workers = 0 if os.name == 'nt' else 8
    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        shuffle=False
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    cosface.load_state_dict(torch.load(r'..\Auxiliary\PretrainedFaceRecognizer\finetuned_cosface.pt'))

    cosface.eval()
    loss, acc = 0.0, 0.0
    for i_batch, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        y_pred = cosface(x)
        loss_batch = loss_fn(y_pred, y)
        # update
        loss += loss_batch.detach().cpu().numpy()
        acc += accuracy(y_pred, y).detach().cpu().numpy()
    loss /= (i_batch + 1)
    acc /= (i_batch + 1)
    print('The test loss is {}, The accuracy is {}'.format(loss, acc))
