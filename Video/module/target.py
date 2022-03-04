import torch
import module.units.arcface_module as arcface_module
import module.units.cosface_module as cosface_module
import module.units.vggface_module as vggface_module
from module.units.facenet_module import InceptionResnetV1, fixed_image_standardization


class ArcFace(object):
    def __init__(self, device, pretrained_path=None, classnum=152):
        self.arcface = arcface_module.ArcFace(classnum=classnum).to(device)
        self.arcface.load_state_dict(torch.load(pretrained_path))
        self.arcface.eval()

    def forward(self, input):
        # costrain input \in [0, 1]
        input = 2.0 * input - 1.0
        logit = self.arcface(input)
        return logit


class CosFace(object):
    def __init__(self, device, pretrained_path=None, classnum=152):
        self.cosface = cosface_module.CosFace(classnum=classnum).to(device)
        self.cosface.load_state_dict(torch.load(pretrained_path))
        self.cosface.eval()

    def forward(self, input):
        # costrain input \in [0, 1]
        input = 2.0 * input - 1.0
        logit = self.cosface(input)
        return logit


class FaceNet(object):

    def __init__(self, device, pretrained_path=None, classnum=152):
        self.resnet = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=classnum).to(device)
        self.resnet.load_state_dict(torch.load(pretrained_path))
        self.resnet.eval()

    def forward(self, input):
        # contrain input \in [0, 1]
        input = input * 255.0
        source = fixed_image_standardization(input)
        logit = self.resnet(source)
        return logit


class VggFace(object):
    def __init__(self, device, pretrained_path=None, classnum=152):
        self.mean = torch.FloatTensor([129.186279296875, 104.76238250732422, 93.59396362304688]).reshape(1, 3, 1, 1).to(device)
        self.vggface = vggface_module.VggFace(embedding_size=4096, classnum=classnum).to(device)
        self.vggface.load_state_dict(torch.load(pretrained_path))
        self.vggface.eval()

    def forward(self, input):
        # contrain input \in [0, 1]
        input = input * 255.0
        input = input - self.mean
        logit = self.vggface(input)
        return logit
