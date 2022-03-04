'''
    Analyzing the influence of stickers on the face detector

'''

import os
import sys
import cv2
import torch
import argparse
from PIL import Image
from mtcnn.mtcnn import MTCNN


def main(args):
    parser = argparse.ArgumentParser(description='Analyzing the influence of stickers on the face detector')
    parser.add_argument('--root', default=r'..\Outputs\AttackingVideos')
    parser.add_argument('--transfer', default=False, action='store_true', help='whether to attack the black model')
    args = parser.parse_args(args)

    sub_dir = None

    if args.transfer:
        sub_dir = 'transfer'
    else:
        sub_dir = 'normal'

    assert os.path.exists(args.root) is True, 'The path should be existed'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(
        image_size=(160, 160), margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        keep_all=True, device=device
    )

    empty = 0.0
    total = 0.0
    num = 0

    for dir_name in os.listdir(args.root):
        sub_dir_path = os.path.join(args.root, os.path.join(dir_name, sub_dir))

        for sub_dir_name in os.listdir(sub_dir_path):
            video_path = os.path.join(sub_dir_path, os.path.join(sub_dir_name, 'unannotated.mp4'))
            cam = cv2.VideoCapture(video_path)

            num += 1
            success, frame = cam.read()
            while success:

                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                faces = mtcnn(frame)
                if faces is None:
                    empty += 1.0

                total += 1.0

                success, frame = cam.read()

            cam.release()

    print('The fraction of images that cannot be detected is {:.4f} %'.format(empty / total * 100))
    print('The main of frames is {}'.format(total / num))


if __name__ == '__main__':
    main(sys.argv[1:])
