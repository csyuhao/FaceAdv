import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from module.units.vggface_module import VggFace


def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()


if __name__ == '__main__':

    random.seed(117)
    np.random.seed(117)
    torch.manual_seed(117)
    torch.cuda.manual_seed(117)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[129.186279296875 / 255.0, 104.76238250732422 / 255.0, 93.59396362304688 / 255.0],
                             std=[1 / 255.0, 1 / 255.0, 1 / 255.0])
    ])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    vggface = VggFace(classnum=156).to(device)
    vggface.load_state_dict(torch.load(r'..\Auxiliary\PretrainedFaceRecognizer\finetuned_vggface.pt'))

    dataset_dir = r'..\Auxiliary\TestFaceBank\lfw-'
    dataset = datasets.ImageFolder(
        dataset_dir, transform=transform)

    batch_size = 8
    loader = DataLoader(
        dataset,
        num_workers=1,
        batch_size=batch_size,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    vggface.eval()
    loss, acc = 0.0, 0.0
    for i_batch, (x, y) in enumerate(loader):
        b = x.shape[0]
        x = x.to(device)
        y = y.to(device)
        y_pred = vggface(x)
        loss_batch = loss_fn(y_pred, y)

        # update
        loss += loss_batch.detach()
        acc += accuracy(y_pred, y).detach()
    loss /= (i_batch + 1)
    acc /= (i_batch + 1)
    print('The test loss is {}, The accuracy is {}'.format(loss, acc))
