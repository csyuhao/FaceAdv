import torch.nn as nn


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.LayerNorm):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class DownsampleConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, padding=1):
        super(DownsampleConvBlock, self).__init__()
        self.norm_layer = nn.LayerNorm(input_size)
        self.relu = nn.ReLU()
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(input_size[0], output_size[0],
                              kernel_size=kernel_size, padding=padding)

    def forward(self, input):
        x = self.norm_layer(input)
        x = self.relu(x)
        x = self.down(x)
        x = self.conv(x)
        return x


class Reshape(nn.Module):

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0), ) + self.shape)


class Discriminator(nn.Module):
    # input_size: 80 * 80 * 1
    def __init__(self, nfc=40):
        super(Discriminator, self).__init__()
        self.ssize = (5, 5)
        self.nfc = nfc
        self.left_branch = self.make_branch(4, (80, 80))
        self.right_branch = self.make_branch(4, (80, 80))
        self.middle_branch = self.make_branch(4, (80, 80))
        self.branchs = [self.left_branch, self.right_branch, self.middle_branch]

    def make_branch(self, nlayers, img_size):
        nfc = self.nfc
        model = [
            nn.Conv2d(1, nfc, kernel_size=3, padding=1)
        ]
        for layer in range(nlayers):
            input_size = [nfc, img_size[0], img_size[1]]
            output_size = [nfc * 2, img_size[0] // 2, img_size[1] // 2]
            model += [DownsampleConvBlock(input_size, output_size)]
            img_size = (img_size[0] // 2, img_size[1] // 2)
            nfc *= 2
        model += [
            nn.LayerNorm(img_size),
            nn.ReLU(),
            Reshape(self.ssize[0] * self.ssize[1] * 16 * self.nfc),
            nn.Linear(self.ssize[0] * self.ssize[1] * 16 * self.nfc, 1)
        ]
        return nn.Sequential(*model)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(m, mean, std)

    # def forward(self, shapes):
    #     n_batch_size = shapes.shape[0] // 3
    #     left, right, middle = shapes[:n_batch_size], shapes[n_batch_size: 2*n_batch_size], shapes[2*n_batch_size:]
    #     left_logits = self.left_branch(left)
    #     right_logits = self.right_branch(right)
    #     middle_logits = self.middle_branch(middle)
    #     logits = (left_logits + right_logits + middle_logits) / 3.0
    #     return logits
