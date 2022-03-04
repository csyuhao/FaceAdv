'''
    Saving the result of video clips for the analysis
        distance : 30cm, 50cm, 70cm
        illuminance: 50lux, 100lux, 150lux
        pose: HN (head normal), HR (head right), HL (head left), HU (head upper), HB (head bottom)

        eyeglasses: AGNs attacking method
        target_class: the victim class or the target class, which deponds on the mode
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
    parser = argparse.ArgumentParser(description='Analysis the Video Clip for Saving Results')
    parser.add_argument('--model', type=str, default='ArcFace')
    parser.add_argument('--target_class', type=int, default=0)
    parser.add_argument('--origin_class', type=int, default=0)
    parser.add_argument('--mode', type=str, default='target')
    parser.add_argument('--distance', type=int, required=True, help='The distance between user and cameras')
    parser.add_argument('--illuminance', type=int, required=True, help='The illuminance degree')
    parser.add_argument('--pose', type=str, required=True, help='The pose of head')
    parser.add_argument('--eyeglasses', default=False, action='store_true', help='whether to wear eyeglasses')
    args = parser.parse_args(args)

    if args.mode == 'target':
        filepath = r'..\Outputs\AttackingVideos\{}_{}_{}'.format(args.model, args.mode, args.target_class)
    else:
        filepath = r'..\Outputs\AttackingVideos\{}_{}_{}'.format(args.model, args.mode, args.origin_class)

    if args.eyeglasses:
        filepath = os.path.join(filepath, 'eyeglasses')
    else:
        filepath = os.path.join(filepath, 'normal')

    subdir = '{}_{}_{}'.format(args.distance, args.illuminance, args.pose)
    filepath = os.path.join(filepath, subdir)

    output_dir = filepath.replace(r'..\Outputs\AttackingVideos', r'..\Outputs\VideoImages')
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cam = cv2.VideoCapture(os.path.join(filepath, 'unannotated.mp4'))

    success, frame = cam.read()

    save_size = (300, 300)

    mtcnn = MTCNN(
        image_size=save_size, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        keep_all=True, device=device
    )

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

    cnt = 0.0
    while success:

        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        save_frame = frame.copy()

        faces = mtcnn(save_frame, save_path='{}/{}.png'.format(output_dir, int(cnt)))
        if faces is None:
            success, frame = cam.read()
            continue

        faces = faces.to(device)
        faces = F.interpolate(faces, size=img_size, mode='bilinear', align_corners=True)
        logit = target_model.forward((faces + 1.0) / 2.0)
        max_id = torch.max(logit, dim=1)[1].cpu().detach().numpy()[0]

        id = args.target_class
        prob = F.softmax(logit, dim=1).cpu().detach().numpy()[0, id]

        with open('results.txt', 'a+') as f:
            f.write('ID:{}, prob :{}, success: {} \n'.format(cnt, prob, max_id != id))
        cnt += 1
        success, frame = cam.read()


if __name__ == '__main__':
    main(sys.argv[1:])
