import torch.nn as nn


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class UpsampleConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, scale_factor=2, kernel_size=3, padding=1):
        super(UpsampleConvBlock, self).__init__()
        self.norm_layer = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=scale_factor,
                              mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(input_channels, output_channels,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, input):
        x = self.norm_layer(input)
        x = self.relu(x)
        x = self.up(x)
        x = self.conv(x)
        return x


class Reshape(nn.Module):

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0), ) + self.shape)


class Generator(nn.Module):
    '''
        output:
            left_sticker  , left_shape  : 3 * 80 * 80, 1 * 80 * 80;
            right_sticker , right_shape : 3 * 80 * 80, 1 * 80 * 80;
            middle_sticker, middle_shape: 3 * 80 * 80, 1 * 80 * 80;
    '''

    def __init__(self, z_dim=32):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.nfc = 16 * 40
        # output size is (self.nfc // 4) * 20 * 20
        self.left_branch = self.make_up_branch(z_dim=z_dim, nlayers=2, factor=self.nfc)
        self.right_branch = self.make_up_branch(z_dim=z_dim, nlayers=2, factor=self.nfc)
        self.middle_branch = self.make_up_branch(z_dim=z_dim, nlayers=2, factor=self.nfc)

        self.branch_left_shape = self.make_branch(nlayers=2, factor=self.nfc / 4, shape=True)
        self.branch_left_sticker = self.make_branch(nlayers=2, factor=self.nfc / 4)

        self.branch_right_shape = self.make_branch(nlayers=2, factor=self.nfc / 4, shape=True)
        self.branch_right_sticker = self.make_branch(nlayers=2, factor=self.nfc / 4)

        self.branch_middle_shape = self.make_branch(nlayers=2, factor=self.nfc / 4, shape=True)
        self.branch_middle_sticker = self.make_branch(nlayers=2, factor=self.nfc / 4)

    def make_up_branch(self, z_dim, nlayers, factor):
        model = [
            nn.Linear(z_dim, factor * 5 * 5),
            nn.BatchNorm1d(factor * 5 * 5),
            nn.ReLU(),
            Reshape(factor, 5, 5)
        ]

        for layer in range(nlayers):
            model += [UpsampleConvBlock(int(factor), int(factor // 2), kernel_size=3)]
            factor /= 2
        return nn.Sequential(*model)

    def make_branch(self, nlayers, factor, shape=False):
        model = []
        for layer in range(nlayers):
            model += [UpsampleConvBlock(int(factor), int(factor // 2), kernel_size=3)]
            factor /= 2

        if shape:
            model += [nn.Conv2d(int(factor), 1, kernel_size=3, padding=1)]
        else:
            model += [nn.Conv2d(int(factor), 3, kernel_size=3, padding=1)]
        model += [nn.Tanh()]
        return nn.Sequential(*model)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(m, mean, std)

    def forward(self, input):
        
        left_branch = self.left_branch(input)
        right_branch = self.right_branch(input)
        middle_branch = self.middle_branch(input)

        # left sticker and left shape
        left_sticker = self.branch_left_sticker(left_branch)
        left_shape = self.branch_left_shape(left_branch)

        # right sticker and right shape
        right_sticker = self.branch_right_sticker(right_branch)
        right_shape = self.branch_right_shape(right_branch)

        # middle sticker and middle shape
        middle_sticker = self.branch_middle_sticker(middle_branch)
        middle_shape = self.branch_middle_shape(middle_branch)

        return [left_sticker, right_sticker, middle_sticker], [left_shape, right_shape, middle_shape]
