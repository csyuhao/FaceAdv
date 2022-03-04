'''
    Getting the result of video
        distance : 30cm, 50cm, 70cm
        illuminance: 50lux, 100lux, 150lux
        pose: HN (head normal), HR (head right), HL (head left), HU (head upper), HB (head bottom)

        eyeglasses: AGNs attacking method
        target_class: the target class
        origin_class: the origin class
        mode: target or untarget
'''

import os
import sys
import cv2
import torch
import argparse
from PIL import Image
from mtcnn.mtcnn import MTCNN
import torch.nn.functional as F
from module.target import ArcFace, CosFace, FaceNet, VggFace


def main(args):
    parser = argparse.ArgumentParser(description='Geneerating Attacking Results')
    parser.add_argument('--model', type=str, default='ArcFace')
    parser.add_argument('--target_class', type=int, default=0)
    parser.add_argument('--origin_class', type=int, default=0)
    parser.add_argument('--mode', type=str, default='target')
    parser.add_argument('--distance', type=int, required=True, help='The distance between user and cameras')
    parser.add_argument('--illuminance', type=int, required=True, help='The illuminance degree')
    parser.add_argument('--pose', type=str, required=True, help='The pose of head')
    parser.add_argument('--transfer', default=False, action='store_true', help='whether to attack the black model')
    args = parser.parse_args(args)

    if args.mode == 'target':
        filepath = r'..\Outputs\AttackingVideos\{}_{}_{}'.format(args.model, args.mode, args.target_class)
    else:
        filepath = r'..\Outputs\AttackingVideos\{}_{}_{}'.format(args.model, args.mode, args.origin_class)

    if args.transfer:
        filepath = os.path.join(filepath, 'transfer')
    else:
        filepath = os.path.join(filepath, 'normal')

    subdir = '{}_{}_{}'.format(args.distance, args.illuminance, args.pose)
    filepath = os.path.join(filepath, subdir)
    assert os.path.exists(filepath) is True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cam = cv2.VideoCapture(os.path.join(filepath, 'unannotated.mp4'))

    success, frame = cam.read()

    classnum = 156
    if args.model == 'ArcFace':
        img_size = (112, 112)
        target_model = ArcFace(device, r'..\Auxiliary\PretrainedFaceRecognizer\finetuned_arcface.pt', classnum)
    elif args.model == 'CosFace':
        img_size = (112, 96)
        target_model = CosFace(device, r'..\Auxiliary\PretrainedFaceRecognizer\finetuned_cosface.pt', classnum)
    elif args.model == 'FaceNet':
        img_size = (160, 160)
        target_model = FaceNet(device, r'..\Auxiliary\PretrainedFaceRecognizer\finetuned_facenet.pt', classnum)
    elif args.model == 'VggFace':
        img_size = (224, 224)
        target_model = VggFace(device, r'..\Auxiliary\PretrainedFaceRecognizer\finetuned_vggface.pt', classnum)
    else:
        raise Exception('This model is not supported.')

    mtcnn = MTCNN(
        image_size=img_size, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        keep_all=True, device=device
    )

    stats = [0.0, 1.0, 0.0, 0.0]
    cnt = 0.0
    while success:

        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        faces = mtcnn(frame)
        if faces is None:
            success, frame = cam.read()
            continue

        faces = faces.to(device)
        logit = target_model.forward((faces + 1.0) / 2.0)

        max_id = torch.max(logit, dim=1)[1].cpu().detach().numpy()[0]
        if args.mode == 'target':
            prob = F.softmax(logit, dim=1).cpu().detach().numpy()[0, args.target_class]
        elif args.mode == 'untarget':
            prob = F.softmax(logit, dim=1).cpu().detach().numpy()[0, args.origin_class]

        if prob > stats[0]:
            stats[0] = prob

        if prob < stats[1]:
            stats[1] = prob

        stats[2] += prob

        if args.mode == 'target':
            stats[3] += float(max_id == args.target_class)
        elif args.mode == 'untarget':
            stats[3] += float(max_id != args.origin_class)

        cnt += 1
        success, frame = cam.read()

    stats[2] /= cnt
    stats[3] /= cnt
    cam.release()
    print('The Maximal Probability:[%.4f], The Minimal Probability: [%.4f], The Mean Probability: [%.4f] and Attack Success: [%.4f]' % (stats[0], stats[1], stats[2], stats[3]))


if __name__ == '__main__':
    main(sys.argv[1:])
