'''
    Collecting VolFace
        distance : 30cm, 50cm, 70cm
        illuminance: 50lux, 100lux, 150lux
        pose: HN (head normal), HR (head right), HL (head left), HU (head upper), HB (head bottom)
'''

import os
import cv2
import sys
import torch
import argparse
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN


def main(args):
    parser = argparse.ArgumentParser(description='Collecting Dataset')
    parser.add_argument('--name', type=str, required=True, help='The name of volunteer')
    parser.add_argument('--num', type=int, default=40)
    parser.add_argument('--distance', type=int, required=True, help='The distance between user and cameras')
    parser.add_argument('--illuminance', type=int, required=True, help='The illuminance degree')
    parser.add_argument('--pose', type=str, required=True, help='The pose of head')
    parser.add_argument('--eyeglasses', default=False, action='store_true', help='whether to wear eyeglasses')

    args = parser.parse_args(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    cam.set(3, 1280)
    cam.set(4, 1024)
    success, frame = cam.read()

    if args.eyeglasses:
        save_path = r'..\Outputs\CollectingDataset\eyeglasses\{}'.format(args.name.replace(' ', '_'))
    else:
        save_path = r'..\Outputs\CollectingDataset\normal\{}'.format(args.name.replace(' ', '_'))
    subdir = '{}_{}_{}'.format(args.distance, args.illuminance, args.pose)
    save_path = os.path.join(save_path, subdir)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    else:
        raise Exception('The directory is already exists.')

    mtcnn = MTCNN(
        image_size=(112, 112), margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        keep_all=True, device=device
    )

    cnt = 0
    cv2.namedWindow('Collecting Dataset')
    while success:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        boxes, _ = mtcnn.detect(frame)
        faces = mtcnn(frame)
        if faces is None:
            cv2.imshow('Collecting Dataset', np.asarray(frame)[..., ::-1])
            success, frame = cam.read()
            continue

        frame.save(os.path.join(save_path, '%s_%2d.png' % (subdir, cnt)))

        frame_draw = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        cv2.imshow('Collecting Dataset', frame_draw)
        cnt += 1
        if cnt >= args.num:
            break
        success, frame = cam.read()

    cam.release()


if __name__ == '__main__':
    main(sys.argv[1:])
