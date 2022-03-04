import enum
import os
import sys
from numpy.testing._private.utils import requires_memory
import torch
import config
import argparse
import numpy as np
import torch.nn as nn
from PIL import Image
import multiprocessing
from scipy import stats
import torch.optim as optim
from dataset import Dataset
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

# load model
from module.generator import Generator
from module.discriminator import Discriminator
from module.target import ArcFace, CosFace, FaceNet, VggFace

# util function
from util import convert_bound, calc_gradient_penalty, attach_stickers_to_image, clipped_sticker_printable, resized_sticker_size


class FaceAdv(object):
    def __init__(self, args):
        self.device = args.device
        self.only_gan = args.only_gan
        self.attack_mode = args.attack_mode
        self.target_class = args.target_class
        self.sticker_size = (90, 90, 90)
        self.attacker_name = args.attacker_name

        self.output_path = args.output_path

        self.attacked_models = args.attacked_models
        self.target_models = {}
        self.img_sizes = {}
        for model_name in args.attacked_models.split(','):

            # Initialize target models
            if model_name == 'ArcFace':
                target_model = ArcFace(self.device)
                img_size = (112, 112)
            elif model_name == 'CosFace':
                target_model = CosFace(self.device)
                img_size = (112, 96)
            elif model_name == 'FaceNet':
                target_model = FaceNet(self.device)
                img_size = (160, 160)
            elif model_name == 'VggFace':
                target_model = VggFace(self.device)
                img_size = (224, 224)
            else:
                raise Exception('The kind of attacked model is unknown...')

            self.target_models[model_name] = target_model
            self.img_sizes[model_name] = img_size

        # Loadding non_printability file
        self.printability_array = nn.Parameter(self.get_printability_array(args.printability_file, config.PRINTABLE_LEFT_STICKER_SIZE),
                                               requires_grad=False).to(self.device)

        # Hyperparameter Settings
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma

        if self.only_gan is True:
            self.alpha = 0.0
            self.beta = 0.0
            self.gamma = 0.0

        # Initialize generator and discriminator
        self.gen = Generator().to(self.device)
        self.gen.weight_init(0., 0.02)
        self.disc = Discriminator().to(self.device)
        self.disc.weight_init(0., 0.02)

        self.optimizer_g = optim.Adam(
            self.gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(
            self.disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

        if args.mode == 'train':
            dir_str = '{}-{}-{}-{}'.format(self.attacked_models.replace(',', '_'), self.attacker_name, self.attack_mode, self.target_class)
            dir_path = os.path.join(args.output_path, dir_str)
            os.makedirs(dir_path, exist_ok=True)
            self.writer = SummaryWriter(dir_path)
            if args.load_pretrained:
                self.pretrained_path = args.pretrained_path
                self.load_pretrained(mode='checkpoint')
        elif args.mode == 'test':
            if args.load_pretrained:
                self.pretrained_path = args.pretrained_path
                self.load_pretrained(mode='test')
            self.gen = self.gen.eval()

    def train(self, base_step, dataset):
        positive = torch.FloatTensor([1]).to(self.device)
        negative = positive * -1
        device = self.device

        xk = np.arange(3)
        pk = [1/3, 1/3, 1/3]
        custm = stats.rv_discrete('custm', values=(xk, pk))

        best_success_rate, best_gen_state_dict, best_distance, best_disc_state_dict = 0., {}, 0., {}
        for ite, batch in enumerate(dataset):

            batch_device = [v.to(device) for v in batch]
            image = batch_device[0]
            shape_real = batch_device[1]
            sticker_config = batch_device[2:]

            image = convert_bound(image, -3)
            shape_real = convert_bound(shape_real, -3)
            batch_size, _, _, _ = image.shape

            # ----------------------- Train D -----------------------
            for p in self.disc.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()
            noise = torch.randn(batch_size, config.NOISE_DIM).to(self.device)

            _, shapes = self.gen(noise)
            indice = custm.rvs(size=1)[0]
            shape_fake = shapes[indice]

            shape_real_v = autograd.Variable(shape_real, requires_grad=True)
            shape_fake_v = autograd.Variable(shape_fake, requires_grad=True)
            # train with real data
            d_real = self.disc.branchs[indice](shape_real_v)
            d_real = d_real.mean(0).view(1)
            d_real.backward(negative)

            # train with fake data
            d_fake = self.disc.branchs[indice](shape_fake_v)
            d_fake = d_fake.mean(0).view(1)
            d_fake.backward(positive)

            gradient_penalty = calc_gradient_penalty(self.disc.branchs[indice], shape_real_v.data, shape_fake_v.data)
            gradient_penalty.backward()

            Wassertein_D = d_real - d_fake

            if Wassertein_D < best_distance:
                best_distance = Wassertein_D
                best_disc_state_dict = self.disc.state_dict()

            self.optimizer_d.step()

            # -------------------- Train G -----------------------
            if ite % config.ITE == 0:

                for p in self.disc.parameters():
                    p.requires_grad = False

                self.optimizer_g.zero_grad()
                noise = torch.randn(batch_size, config.NOISE_DIM).to(self.device)
                noise_v = autograd.Variable(noise, requires_grad=True)
                image_v = autograd.Variable(image, requires_grad=True)

                stickers, shapes = self.gen(noise_v)

                shape_fake = shapes[indice]
                g_fake = self.disc.branchs[indice](shape_fake)

                left_sticker, right_sticker, middle_sticker = stickers[0], stickers[1], stickers[2]
                mask, clipped_sticker, image_with_sticker = attach_stickers_to_image(left_sticker, shapes[0],
                                                                                     right_sticker, shapes[1],
                                                                                     middle_sticker, shapes[2],
                                                                                     image_v, sticker_config, self.sticker_size, mode='train')

                resized_image, g_clf, g_prob, g_success_rate = self.calc_classification_loss(image_with_sticker)

                resized_left_sticker, resized_right_sticker, resized_middle_sticker = resized_sticker_size(left_sticker, right_sticker,
                                                                                                           middle_sticker, self.sticker_size)
                printed_stickers = torch.cat([resized_left_sticker, resized_middle_sticker, resized_right_sticker], dim=0)
                g_tv_loss = self.clac_tv_loss(printed_stickers)

                g_non_printability_loss, cnt = 0., 0.0
                for _, sticker in enumerate(printed_stickers):
                    g_non_printability_loss += self.non_printability_loss(sticker.unsqueeze(0))
                    cnt += 1.0
                g_non_printability_loss /= cnt

                G_cost = -g_fake.mean() + self.alpha * g_clf.mean() + self.beta * g_tv_loss.mean() + self.gamma * g_non_printability_loss.mean()
                G_cost.backward()
                self.optimizer_g.step()

                if best_success_rate < g_success_rate.detach().cpu().item():
                    best_success_rate = g_success_rate.detach().cpu().item()
                    best_gen_state_dict = self.gen.state_dict()

            if ite % 20 == 0:
                # images
                self.writer.add_image('1_image_with_sticker', image_with_sticker[0].detach(), base_step + ite)
                self.writer.add_image('2_clipped_sticker', clipped_sticker[0].detach(), base_step + ite)
                self.writer.add_image('3_mask_with_different_shapes', mask[0].detach(), base_step + ite)
                self.writer.add_image('4_resized_image', resized_image[0].detach(), base_step + ite)
                # scalars
                self.writer.add_scalar('1_g_prob', g_prob.detach(), base_step + ite)
                self.writer.add_scalar('2_g_success_rate', g_success_rate.detach(), base_step + ite)
                self.writer.add_scalar('3_d_Wassertein_D', Wassertein_D.detach(), base_step + ite)
                self.writer.add_scalar('4_g_clf', g_clf.mean().detach(), base_step + ite)
                self.writer.add_scalar('5_g_tv_loss', g_tv_loss.mean().detach(), base_step + ite)
                self.writer.add_scalar('6_g_non_printability_loss', g_non_printability_loss.mean().detach(), base_step + ite)

        self.save_model(best_disc_state_dict, best_gen_state_dict)

    def calc_classification_loss(self, image):
        batch_size = image.shape[0]
        # classification loss
        labels = torch.LongTensor([self.target_class] * batch_size).to(self.device)

        cnt = 0.
        mean_clf_loss, mean_prob, mean_success_rate = 0., 0., 0.
        for model_name in self.attacked_models.split(','):
            resized_image = F.interpolate(image, size=self.img_sizes[model_name], mode='bilinear', align_corners=True)
            logits = self.target_models[model_name].forward(resized_image)
            if self.attack_mode == 'target':
                success_rate = (torch.max(logits, dim=1)[1] == labels).float().mean()
                c_loss = F.cross_entropy(logits, labels)
                prob = (F.softmax(logits, dim=1)).gather(1, labels.view(batch_size, -1)).mean()
            elif self.attack_mode == 'untarget':
                success_rate = 1 - (torch.max(logits, dim=1)[1] == labels).float().mean()
                c_loss = - F.cross_entropy(logits, labels)
                prob = (F.softmax(logits, dim=1)).gather(1, labels.view(batch_size, -1)).mean()
            else:
                raise NotImplementedError('The attack mode is undefined')
            mean_success_rate += success_rate
            mean_clf_loss += c_loss
            mean_prob += prob
            cnt += 1.0

        return resized_image, mean_clf_loss / cnt, mean_prob / cnt, mean_success_rate / cnt

    @torch.no_grad()
    def calc_prob(self, image):
        batch_size = image.shape[0]
        # classification loss
        labels = torch.LongTensor([self.target_class] * batch_size).to(self.device)

        prob, cnt, success_rate = 0.0, 0.0, 0.0
        for model_name in self.attacked_models.split(','):
            resized_image = F.interpolate(image, size=self.img_sizes[model_name], mode='bilinear', align_corners=True)
            logits = self.target_models[model_name].forward(resized_image)

            if self.attack_mode == 'target':
                success_rate += (torch.max(logits, dim=1)[1] == labels).float().mean()
                prob += (F.softmax(logits, dim=1)).gather(1, labels.view(batch_size, -1)).mean()
            elif self.attack_mode == 'untarget':
                success_rate += 1 - (torch.max(logits, dim=1)[1] == labels).float().mean()
                prob += (F.softmax(logits, dim=1)).gather(1, labels.view(batch_size, -1)).mean()

            cnt += 1.0

        return prob / cnt, success_rate / cnt

    def clac_tv_loss(self, img):
        img = convert_bound(img, mode=3)
        batch_size = img.size()[0]
        h_img = img.size()[2]
        w_img = img.size()[3]
        count_h = (img.size()[2] - 1) * img.size()[3]
        count_w = img.size()[2] * (img.size()[3] - 1)
        h_tv = torch.pow((img[:, :, 1:, :] - img[:, :, :h_img - 1, :]), 2).reshape((batch_size, -1)).sum(dim=1, keepdim=True)
        w_tv = torch.pow((img[:, :, :, 1:] - img[:, :, :, :w_img - 1]), 2).reshape((batch_size, -1)).sum(dim=1, keepdim=True)
        return 2 * (h_tv / count_h + w_tv / count_w)

    def non_printability_loss(self, adv_stickers):
        adv_stickers = convert_bound(adv_stickers, mode=3)
        # Calculating euclidian distance between colors in adversarial stickers and colors in printablity arrys
        color_dist = (adv_stickers - self.printability_array + 1e-6)
        color_dist = color_dist**2
        color_dist = torch.sum(color_dist, 1) + 1e-6
        color_dist = torch.sqrt(color_dist)

        # Only work with the minimal distance
        color_dist_prod = torch.min(color_dist, 0)[0]

        # Caculating the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_stickers)

    def get_printability_array(self, printability_file, size):

        printability_list = []

        # Read in printability triplets and put them in a list
        with open(printability_file, 'r') as f:
            for line in f.readlines():
                printability_list.append(line.split(','))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((size, size), red))
            printability_imgs.append(np.full((size, size), green))
            printability_imgs.append(np.full((size, size), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asfarray(printability_array, dtype=np.float)
        pa = torch.from_numpy(printability_array)
        return pa

    def save_model(self, disc_state_dict, gen_state_dict):
        dir_str = '{}-{}-{}-{}'.format(self.attacked_models.replace(',', '_'), self.attacker_name, self.attack_mode, self.target_class)
        dir_path = os.path.join(self.output_path, dir_str)
        os.makedirs(dir_path, exist_ok=True)

        # save generator model
        model_save_path = os.path.join(dir_path, 'gen_best_epoch.pt')
        torch.save(gen_state_dict, model_save_path)
        # save discriminator model
        model_save_path = os.path.join(dir_path, 'disc_best_epoch.pt')
        torch.save(disc_state_dict, model_save_path)

    def load_pretrained(self, mode='checkpoint'):
        print('-------- load pretrained model parameters -------')
        if mode == 'checkpoint':
            # Reloading generator model
            model_save_path = os.path.join(self.pretrained_path, 'gen_best_epoch.pt')
            print('Pre-trained generator model is {0}'.format(model_save_path))
            self.gen.load_state_dict(torch.load(model_save_path))
            # Reloading discriminator model
            model_save_path = os.path.join(self.pretrained_path, 'disc_best_epoch.pt')
            print('Pre-trained discriminator model is {0}'.format(model_save_path))
            self.disc.load_state_dict(torch.load(model_save_path))
        elif mode == 'test':
            # Reloading generator model
            model_save_path = os.path.join(self.pretrained_path, 'gen_best_epoch.pt')
            print('Pretrained generator model is {0}'.format(model_save_path))
            self.gen.load_state_dict(torch.load(model_save_path))
        print('-------- load finished---------')

    @torch.no_grad()
    def test(self, testing_dataset):

        sample_iter = config.SAMPLE_ITER
        best_noise = None
        best_prob = 0.0 if self.attack_mode == 'target' else 1.0
        best_success_rate = 0.0
        best_image_with_sticker = None

        for _ in range(sample_iter):
            start_noise = torch.randn(1, config.NOISE_DIM).to(self.device)
            g_prob, g_success_rate, cnt = 0.0, 0.0, 0.0
            for _, batch in enumerate(testing_dataset):
                batch_device = [v.to(self.device) for v in batch]
                image = batch_device[0]
                image = 2.0 * image - 1.0
                sticker_config = batch_device[1:]

                batch_size, _, _, _ = image.shape
                noise = start_noise.repeat(batch_size, 1)
                [left_sticker, right_sticker, middle_sticker], [left_shape, right_shape, middle_shape] = self.gen(noise)
                _, _, image_with_sticker = attach_stickers_to_image(left_sticker, left_shape,
                                                                    right_sticker, right_shape,
                                                                    middle_sticker, middle_shape,
                                                                    image, sticker_config, self.sticker_size, mode='test')
                prob, success_rate= self.calc_prob(image_with_sticker)
                g_prob += prob.detach().cpu().item()
                g_success_rate += success_rate.detach().cpu().item()
                cnt += 1.0
            g_prob /= cnt
            g_success_rate /= cnt

            if self.attack_mode == 'target' and g_prob > best_prob:
                best_prob = g_prob
                best_success_rate = g_success_rate
                best_noise = start_noise.clone().detach()
                best_image_with_sticker = image_with_sticker.clone().detach()
            elif self.attack_mode == 'untarget' and g_prob < best_prob:
                best_prob = g_prob
                best_success_rate = g_success_rate
                best_noise = start_noise.clone().detach()
                best_image_with_sticker = image_with_sticker.clone().detach()

        [left_sticker, right_sticker, middle_sticker], [left_shape, right_shape, middle_shape] = self.gen(best_noise)
        clipped_left_sticker, clipped_right_sticker, clipped_middle_sticker = clipped_sticker_printable(left_sticker, left_shape,
                                                                                                        right_sticker, right_shape,
                                                                                                        middle_sticker, middle_shape,
                                                                                                        self.sticker_size)

        # save images
        save_dir = '{}-{}-{}-{}'.format(self.attacked_models.replace(',', '_'), self.attacker_name, self.attack_mode, self.target_class)
        save_path = os.path.join(self.output_path, save_dir)
        os.makedirs(save_path, exist_ok=True)
        # transform
        trans = transforms.Compose([
            transforms.ToPILImage()
        ])
        # left stickers
        clipped_left_sticker = trans(clipped_left_sticker[0].cpu())
        clipped_left_sticker.save(os.path.join(save_path, 'left sticker.png'), 'PNG')
        # middle stickers
        clipped_right_sticker = trans(clipped_right_sticker[0].cpu())
        clipped_right_sticker.save(os.path.join(save_path, 'right sticker.png'), 'PNG')
        # right stickers
        clipped_middle_sticker = trans(clipped_middle_sticker[0].cpu())
        clipped_middle_sticker.save(os.path.join(save_path, 'middle sticker.png'), 'PNG')
        # info
        with open(os.path.join(save_path, 'info.txt'), 'w+') as f:
            f.write('The maximum prob is {:.4f}, The maximum success rate is {:.4f}'.format(best_prob, best_success_rate))
        # save image with stickers
        save_image(best_image_with_sticker, '{0}/{1}'.format(save_path, 'image_with_sticker.png'), nrow=6, normalize=False)


def train(args):

    image = os.path.join(args.train_dataset, os.path.join('image', args.attacker_name))
    shape = os.path.join(args.train_dataset, 'shape')
    trans = transforms.Compose([
        Image.open,
        transforms.ToTensor(),
    ])
    dataset = Dataset(image, shape, args.locs, transform=trans)
    training_dataset = DataLoader(dataset, batch_size=args.batch_size,
                                  shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=True)
    len_dataset = len(training_dataset)
    faceadv = FaceAdv(args)
    for epoch in range(args.epochs):
        base_step = len_dataset * epoch
        faceadv.train(base_step, training_dataset)


def test(args):

    image = os.path.join(args.test_dataset, os.path.join('image', args.attacker_name))
    trans = transforms.Compose([
        Image.open,
        transforms.ToTensor(),
    ])
    dataset = Dataset(image, None, args.locs, transform=trans)
    testing_dataset = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    faceadv = FaceAdv(args)
    faceadv.test(testing_dataset)


def main(argv):

    # Fast Training
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='The mian file for training and testing')
    parser.add_argument('--train_dataset', default=r'..\Auxiliary\TrainGeneratorDataset', type=str, help='The path of the training dataset')
    parser.add_argument('--test_dataset', default=r'..\Auxiliary\TrainGeneratorDataset', type=str, help='The path of the testing dataset')
    parser.add_argument('--attacker_name', default=r'NAME', type=str, help='The name of the attacker')
    parser.add_argument('--mode', default='train', type=str, help='The running mode')
    parser.add_argument('--epochs', default=1, type=int, help='The number of epochs')
    parser.add_argument('--device', default='cuda', type=str, help='Which deivce (cpu or cuda)')
    parser.add_argument('--lr', default=5e-4, type=float, help='The learning rate')
    parser.add_argument('--only_gan', default=False, type=bool, help='Whether to train gan only')
    parser.add_argument('--alpha', default=1e2, type=float, help='The weight of classfication loss')
    parser.add_argument('--beta', default=1.0, type=float, help='The weight of tv loss')
    parser.add_argument('--gamma', default=10.0, type=float, help='The weight of non_printability loss')
    parser.add_argument('--attack_mode', default='target', type=str, help='The attack mode')
    parser.add_argument('--attacked_models', default='FaceNet', type=str, help='The attcked model (ArcFace, CosFace, FaceNet)')
    parser.add_argument('--target_class', default=1, type=int, help='The target class of classification')
    parser.add_argument('--batch_size', default=32, type=int, help='The batch size')
    parser.add_argument('--shuffle', default=True, action='store_true', help='Whether to shuffle dataset or not')
    parser.add_argument('--num_workers', default=1, type=int, help='The number of loading dataset')
    parser.add_argument('--locs', default=1, type=int, help='The id of combination of stickers')
    parser.add_argument('--load_pretrained', default=False, action='store_true', help='Whether to load pretrained')
    parser.add_argument('--pretrained_path', default=r'..\Auxiliary\Checkpoints', type=str, help='The path of the pretrained models')
    parser.add_argument('--output_path', default=r'..\Outputs\LogsStickersAndModels', type=str, help='The dir of output in testing')
    parser.add_argument('--printability_file', default=r'..\Auxiliary\Configurations\printability.txt', type=str, help='The printability file')
    args = parser.parse_args(argv)
    if args.mode == 'train':
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    if os.name == 'nt':
        multiprocessing.freeze_support()
    main(sys.argv[1:])
