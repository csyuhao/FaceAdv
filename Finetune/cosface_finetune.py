import os
import torch
import random
import numpy as np
import torch.optim as optim
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

    cosface = CosFace(classnum=156, pretrained=r'..\Auxiliary\PretrainedFeatureExtractor\ACC99.28.pth').to(device)

    dataset_dir = r'..\Auxiliary\ClippedFaceBank'
    dataset = datasets.ImageFolder(
        dataset_dir, transform=transform)
    len_imgs = int(len(dataset) * 0.2)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - len_imgs, len_imgs])

    batch_size = 32
    workers = 0 if os.name == 'nt' else 8
    epochs = 20
    train_loader = DataLoader(
        train_dataset,
        num_workers=workers,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=workers,
        batch_size=1,
        shuffle=False
    )

    optimizer = optim.Adam(cosface.logits.parameters(), lr=1e-3)

    loss_fn = torch.nn.CrossEntropyLoss()

    cosface.backbone.eval()

    best_acc, best_state_dict = 0., {}
    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        cosface.logits.train()
        loss = 0.0
        acc = 0.0
        for i_batch, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = cosface(x)
            loss_batch = loss_fn(y_pred, y)
            # update
            loss_batch.backward()
            optimizer.step()
            loss += loss_batch.detach().cpu().numpy()
            acc += accuracy(y_pred, y).detach().cpu().numpy()
        loss /= (i_batch + 1)
        acc /= (i_batch + 1)
        print('The train loss is {}, The accuracy is {}'.format(loss, acc))

        cosface.logits.eval()
        loss, acc = 0.0, 0.0
        for i_batch, (x, y) in enumerate(test_loader):
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

        if best_acc < acc:
            best_acc = acc
            best_state_dict = cosface.state_dict()

    os.makedirs(r'..\Auxiliary\PretrainedFaceRecognizer', exist_ok=True)
    torch.save(best_state_dict, r'..\Auxiliary\PretrainedFaceRecognizer\finetuned_cosface.pt')
