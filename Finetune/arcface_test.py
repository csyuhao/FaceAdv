import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from module.units.arcface_module import ArcFace


def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()


if __name__ == '__main__':

    random.seed(117)
    np.random.seed(117)
    torch.manual_seed(117)
    torch.cuda.manual_seed(117)

    transform = transforms.Compose([
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    arcface = ArcFace(classnum=156).to(device)
    arcface.load_state_dict(torch.load(r'..\Auxiliary\PretrainedFaceRecognizer\finetuned_arcface.pt'))

    dataset_dir = r'..\Auxiliary\TestFaceBank\lfw-'
    dataset = datasets.ImageFolder(
        dataset_dir, transform=transform)

    batch_size = 16
    loader = DataLoader(
        dataset,
        num_workers=1,
        batch_size=batch_size,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    arcface.eval()
    loss, acc = 0.0, 0.0
    for i_batch, (x, y) in enumerate(loader):
        b = x.shape[0]
        x = x.to(device)
        y = y.to(device)
        y_pred = arcface(x)
        loss_batch = loss_fn(y_pred, y)

        # update
        loss += loss_batch.detach()
        acc += accuracy(y_pred, y).detach()
    loss /= (i_batch + 1)
    acc /= (i_batch + 1)
    print('The test loss is {}, The accuracy is {}'.format(loss, acc))
