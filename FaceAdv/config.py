IMAGE_SIZE = 300
NOISE_DIM = 32
ITE = 1
CLASS_NUM = 153
PRINTABLE_LEFT_STICKER_SIZE = 170
PRINTABLE_MIDDLE_STICKER_SIZE = 170
LAMB = 10.0         # gradient penalty lambda
MINIMAL = 0.2       # clipping sticker for specific shape
SAMPLE_ITER = 64

# --------------- ArcFace Model Settings ----------
ARCFACE_PRETRAINED_PATH = r'..\Auxiliary\PretrainedFaceRecognizer\finetuned_arcface.pt'


# --------------- CosFace Model Settings ----------
COSFACE_PRETRAINED_PATH = r'..\Auxiliary\PretrainedFaceRecognizer\finetuned_cosface.pt'


# --------------- FaceNet Model Settings ----------
FACENET_PRETRAINED = 'vggface2'
FACENET_PRETRAINED_PATH = r'..\Auxiliary\PretrainedFaceRecognizer\finetuned_facenet.pt'

# --------------- VGGFACE Model Settings ----------
VGGFACE_PRETRAINED_PATH = r'..\Auxiliary\PretrainedFaceRecognizer\finetuned_vggface.pt'
