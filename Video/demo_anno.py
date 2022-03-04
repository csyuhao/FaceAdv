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
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from module.target import ArcFace, CosFace, FaceNet, VggFace


def main():

    fpath = r'E:\CurrentResearch\FaceAdv_Components\FaceAdv_Repair\Outputs\DemoVideos\Wei_Ya_Qian.mp4'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cam = cv2.VideoCapture(fpath)

    success, frame = cam.read()

    classnum = 156
    img_size = (160, 160)
    target_model = FaceNet(device, r'..\Auxiliary\PretrainedFaceRecognizer\finetuned_facenet.pt', classnum)

    mtcnn = MTCNN(
        image_size=img_size, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        keep_all=True, device=device
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 1
    size = (1280, 960)
    annotated_out = cv2.VideoWriter(os.path.join(r'E:\CurrentResearch\FaceAdv_Components\FaceAdv_Repair\Outputs\DemoVideos', 'Wei_Ya_Qian_annotated.mp4'), fourcc, fps, size)
    font = ImageFont.truetype("consola.ttf", 36, encoding="unic")

    name_list = r'E:\CurrentResearch\FaceAdv_Components\FaceAdv_Repair\Auxiliary\Name.list'
    names = []
    with open(name_list, 'r') as f:
        for line in f.readlines():
            names.append(line.strip())

    while success:

        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        boxes, _ = mtcnn.detect(frame)
        faces = mtcnn(frame)
        if faces is None:
            success, frame = cam.read()
            continue

        faces = faces.to(device)
        logit = target_model.forward((faces + 1.0) / 2.0)

        max_id = torch.max(logit, dim=1)[1].cpu().detach().numpy()[0]
        prob = F.softmax(logit, dim=1).cpu().detach().numpy()[0, max_id]

        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        for box in boxes[:1]:
            draw.text((box.tolist()[0], box.tolist()[3] + 10), 'Id: %s \n Conf: %.4f' % (names[max_id], prob), (255, 0, 0), font=font)
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

        frame_draw = cv2.cvtColor(np.asarray(frame_draw), cv2.COLOR_RGB2BGR)

        annotated_out.write(frame_draw)

        success, frame = cam.read()

    cam.release()
    annotated_out.release()


if __name__ == '__main__':
    main()
