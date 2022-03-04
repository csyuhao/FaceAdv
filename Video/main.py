'''
    Attacking Face Recognition Systems for recording videos
        distance : 30cm, 50cm, 70cm
        illuminance: 50lux, 100lux, 150lux
        pose: HN (head normal), HR (head right), HL (head left), HU (head upper), HB (head bottom)

        eyeglasses: AGNs attacking method
        target_class: the victim class or the target class, which deponds on the mode
'''

import os
import sys
import cv2
import time
import torch
import argparse
import numpy as np
from mtcnn.mtcnn import MTCNN
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from module.target import ArcFace, CosFace, FaceNet, VggFace


def main(args):
    parser = argparse.ArgumentParser(description='Recording Attacking Video')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--target_class', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--distance', type=int, required=True, help='The distance between user and cameras')
    parser.add_argument('--illuminance', type=int, required=True, help='The illuminance degree')
    parser.add_argument('--pose', type=str, required=True, help='The pose of head')
    parser.add_argument('--transfer', default=False, action='store_true', help='whether to attack the black model')
    args = parser.parse_args(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cam.set(3, 1280)
    cam.set(4, 1024)
    cv2.namedWindow('Attacking Face Recognition System')
    success, frame = cam.read()
    saved_frame = frame.copy()

    save_path = r'..\Outputs\AttackingVideos\{}_{}_{}'.format(args.model, args.mode, args.target_class)
    if args.transfer:
        save_path = os.path.join(save_path, 'transfer')
    else:
        save_path = os.path.join(save_path, 'normal')
    subdir = '{}_{}_{}'.format(args.distance, args.illuminance, args.pose)
    save_path = os.path.join(save_path, subdir)
    assert os.path.exists(save_path) is False
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 1
    size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    annotated_out = cv2.VideoWriter(os.path.join(save_path, 'annotated.mp4'), fourcc, fps, size)
    unannotated_out = cv2.VideoWriter(os.path.join(save_path, 'unannotated.mp4'), fourcc, fps, size)

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
    font = ImageFont.truetype("consola.ttf", 18, encoding="unic")

    start_time = time.time()
    cnt = 0
    while success:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        boxes, _ = mtcnn.detect(frame)
        faces = mtcnn(frame)
        cnt += 1
        if faces is None:
            cv2.imshow('Attacking Face Recognition System', np.asarray(frame)[..., ::-1])
            end_time = time.time()
            if (end_time - start_time) > 20:
                break
            success, frame = cam.read()
            continue

        faces = faces.to(device)
        logit = target_model.forward((faces + 1.0) / 2.0)

        id = args.target_class
        prob = F.softmax(logit, dim=1).cpu().detach().numpy()[0, id]

        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        for box in boxes:
            draw.text((box.tolist()[0], box.tolist()[1] - 20), 'Id: %d Conf: %.4f' % (id, prob), (255, 0, 0), font=font)
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

        frame_draw = cv2.cvtColor(np.asarray(frame_draw), cv2.COLOR_RGB2BGR)

        annotated_out.write(frame_draw)
        unannotated_out.write(saved_frame)

        cv2.imshow('Attacking Face Recognition System', frame_draw)
        end_time = time.time()
        if (end_time - start_time) > 25:
            break
        success, frame = cam.read()
        saved_frame = frame.copy()

    cam.release()
    annotated_out.release()
    unannotated_out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1:])
