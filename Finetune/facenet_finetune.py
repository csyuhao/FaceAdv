# reference https://github.com/timesler/facenet-pytorch/blob/master/examples/finetune.ipynb
import os
import torch
import random
import numpy as np
from torch import optim
from module import training
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR
from module.units.facenet_module import InceptionResnetV1, fixed_image_standardization


if __name__ == '__main__':

    random.seed(117)
    np.random.seed(117)
    torch.manual_seed(117)
    torch.cuda.manual_seed(117)

    data_dir = r'..\Auxiliary\ClippedFaceBank'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    # prepare datasets
    trans = transforms.Compose([
        transforms.Resize((160, 160)),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    dataset_dir = r'..\Auxiliary\TestFaceBank\lfw-'
    dataset = datasets.ImageFolder(
        dataset_dir, transform=trans)
    print('the size of dataset is {}'.format(len(dataset)))
    len_imgs = len(dataset) // 10
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
        batch_size=batch_size,
        shuffle=False
    )

    # model
    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=len(dataset.class_to_idx)
    ).to(device)

    print('the size of dataset class is {}'.format(len(dataset.class_to_idx)))
    optimizer = optim.Adam(resnet.logits.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, [60, 80])

    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
    }

    print('\n\nInitial')
    print('-' * 10)

    resnet.eval()

    best_acc, best_state_dict = 0., {}

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        resnet.logits.train()
        training.pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=None, training=True
        )

        resnet.logits.eval()
        _, m = training.pass_epoch(
            resnet, loss_fn, test_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=None, training=False
        )

        if best_acc < m['acc'].item():
            best_acc = m['acc'].item()
            best_state_dict = resnet.state_dict()

    os.makedirs(r'..\Auxiliary\PretrainedFaceRecognizer', exist_ok=True)
    torch.save(best_state_dict, r'..\Auxiliary\PretrainedFaceRecognizer\finetuned_facenet.pt')
