import os
import glob
import torch
import random
import numpy as np
from scipy.io import loadmat, savemat
from models.resnet_50 import resnet50_use
from load_data import transfer_BFM09, BFM, load_img, Preprocess, transfer_UV
from reconstruction_mesh import reconstruction, transform_face_shape, estimate_intrinsic


candidate_locs = [1, 2, 3, 4, 5]
candidate_names = ['Left-Superciliary-Arch', 'Right-Superciliart-Arch', 'Nasal-Bone', 'Left-Nasolabial-Sulcus', 'Right-Nasolabial-Sulcus']

candidate_groups = [
    [1, 2, 3],
    [1, 2, 4],
    [1, 2, 5],
    [1, 3, 4],
    [1, 3, 5],
    [1, 4, 5],
    [2, 3, 4],
    [2, 3, 5],
    [2, 4, 5],
    [3, 4, 5]
]


def location(idx, landmarks):
    if idx == 1:
        diff = landmarks[37, 1] - landmarks[19, 1] - 10
        loc = [landmarks[19, 0], landmarks[19, 1] + diff]
        return loc
    elif idx == 2:
        diff = landmarks[43, 1] - landmarks[24, 1] - 10
        loc = [landmarks[24, 0], landmarks[24, 1] + diff]
        return loc
    elif idx == 3:
        diff = landmarks[30, 1] - landmarks[29, 1] - 10
        loc = [landmarks[29, 0], landmarks[29, 1] + diff]
        return loc
    elif idx == 4:
        return [int((landmarks[19, 0] + landmarks[20, 0]) / 2.0), int((landmarks[34, 1] + landmarks[51, 1]) / 2.0) + 45]
    elif idx == 5:
        return [int((landmarks[23, 0] + landmarks[24, 0]) / 2.0), int((landmarks[34, 1] + landmarks[51, 1]) / 2.0) + 45]


def recon():
    # input and output folder
    image_path = r'dataset'
    save_path = r'output'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_list = glob.glob(image_path + '/**/' + '*.png', recursive=True)
    img_list += glob.glob(image_path + '/**/' + '*.jpg', recursive=True)

    # read BFM face model
    # transfer original BFM model to our model
    if not os.path.isfile(r'BFM\BFM_model_front.mat'):
        transfer_BFM09()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    bfm = BFM(r'BFM/BFM_model_front.mat', device)

    # read standard landmarks for preprocessing images
    lm3D = bfm.load_lm3d()

    model = resnet50_use().to(device)
    model.load_state_dict(torch.load(r'models\params.pt'))
    model.eval()

    for file in img_list:
        # load images and corresponding 5 facial landmarks
        img, lm = load_img(file, file.replace('.jpg', '.txt').replace('.png', '.txt'))

        # preprocess input image
        input_img_org, lm_new, transform_params = Preprocess(img, lm, lm3D)

        input_img = input_img_org.astype(np.float32)
        input_img = torch.from_numpy(input_img).permute(0, 3, 1, 2)
        # the input_img is BGR
        input_img = input_img.to(device)

        with torch.no_grad():
            arr_coef = model(input_img)

        coef = torch.cat(arr_coef, 1)

        # reconstruct 3D face with output coefficients and face model
        face_shape, face_texture, face_color, landmarks_2d, z_buffer, angles, translation, gamma = reconstruction(coef, bfm)

        fx, px, fy, py = estimate_intrinsic(landmarks_2d, transform_params, z_buffer, face_shape, bfm, angles, translation)

        face_shape_t = transform_face_shape(face_shape, angles, translation)
        face_color = face_color / 255.0
        face_shape_t[:, :, 2] = 10.0 - face_shape_t[:, :, 2]

        path_str = file.replace(image_path, save_path)
        path = os.path.split(path_str)[0]
        if os.path.exists(path) is False:
            os.makedirs(path)

        # loading key points
        tex_coords = transfer_UV()
        tex_size = 600.0
        tex_coords = tex_size * tex_coords
        tex_coords[:, 1] = 600.0 - tex_coords[:, 1]
        front_model = loadmat(r'BFM\BFM_model_front.mat')
        keypoints = front_model['keypoints'].astype(np.int32)
        landmarks = tex_coords[keypoints][0]

        locations = {}
        for idx, candidate_group in enumerate(candidate_groups):
            left_idx, right_idx, middle_idx = candidate_group
            left_loc, right_loc, middle_loc = location(left_idx, landmarks), location(right_idx, landmarks), location(middle_idx, landmarks)

            locations['locs_{}'.format(idx + 1)] = np.array([left_loc, right_loc, middle_loc])

        fx = fx.detach().cpu().numpy()
        px = px.detach().cpu().numpy()
        fy = fy.detach().cpu().numpy()
        py = py.detach().cpu().numpy()

        gamma = gamma.squeeze(0).detach().cpu().numpy()

        face_shape_t = face_shape_t.squeeze(0).detach().cpu().numpy()

        triangle = bfm.tri

        tex_coords = transfer_UV()

        savemat(path_str.replace('.png', '.mat'), dict({
            'face': face_shape_t,
            'tri': triangle,
            'gamma': gamma,
            'intrinsic': [fx, px, fy, py],
            'uv': tex_coords,
        }, **locations))


if __name__ == '__main__':

    random.seed(117)
    np.random.seed(117)
    torch.manual_seed(117)
    torch.cuda.manual_seed(117)

    recon()
