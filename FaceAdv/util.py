import math
import torch
import config
import torch.nn.functional as F
import torch.autograd as autograd
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
from pytorch3d.renderer import (
    SfMPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
)
from module.render.shader import TexturedSoftPhongShader


def resized_sticker_size(left_sticker, right_sticker, middle_sticker, sticker_size):
    resized_left_sticker = F.interpolate(left_sticker, size=config.PRINTABLE_LEFT_STICKER_SIZE, mode='bilinear', align_corners=True)
    resized_right_sticker = F.interpolate(right_sticker, size=config.PRINTABLE_LEFT_STICKER_SIZE, mode='bilinear', align_corners=True)
    resized_middle_sticker = F.interpolate(middle_sticker, size=config.PRINTABLE_MIDDLE_STICKER_SIZE, mode='bilinear', align_corners=True)
    return resized_left_sticker, resized_right_sticker, resized_middle_sticker


def convert_bound(input, mode):
    # convert values of matrix to standard range
    if mode == 1:
        # covert [0, 255] to [0, 1]
        input = input / 255.0
    elif mode == 2:
        # covert [0, 255] to [-1, 1]
        input = 2.0 * (input / 255.0) - 1.0
    elif mode == 3:
        # convert [-1, 1] to [0, 1]
        input = (input + 1.0) / 2.0
    elif mode == -1:
        # covert [0, 1] to [0, 255]
        input = input * 255.0
    elif mode == -2:
        # convert [-1, 1] to [0, 255]
        input = (input + 1.0) / 2.0 * 255.0
    elif mode == -3:
        # convert [0, 1] to [-1, 1]
        input = (input * 2.0) - 1.0
    else:
        raise NotImplementedError(
            'convert mode [%d] is not implemented' % (mode))
    return input


def calc_gradient_penalty(netD, real_data, fake_data):
    batch_size, c, h, w = real_data.shape
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_data.nelement() // batch_size).contiguous().view(batch_size, c, h, w)
    alpha = alpha.to(real_data.device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(real_data.device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean(dim=0, keepdim=True) * config.LAMB
    return gradient_penalty


def clipped_sticker_printable(left_sticker, left_shape_fake, right_sticker, right_shape_fake, middle_sticker, middle_shape_fake, sticker_size):
    # convert [-1, 1] to [0, 1]
    left_sticker = convert_bound(left_sticker, 3)
    left_shape_fake = convert_bound(left_shape_fake, 3)
    right_sticker = convert_bound(right_sticker, 3)
    right_shape_fake = convert_bound(right_shape_fake, 3)
    middle_sticker = convert_bound(middle_sticker, 3)
    middle_shape_fake = convert_bound(middle_shape_fake, 3)
    # size rescaling
    output_size = int(config.PRINTABLE_LEFT_STICKER_SIZE / 90 * sticker_size[0])
    left_sticker = F.interpolate(left_sticker, size=(output_size, output_size), mode='bilinear', align_corners=True)
    left_shape_fake = F.interpolate(left_shape_fake, size=(output_size, output_size), mode='bilinear', align_corners=True)
    right_sticker = F.interpolate(right_sticker, size=(output_size, output_size), mode='bilinear', align_corners=True)
    right_shape_fake = F.interpolate(right_shape_fake, size=(output_size, output_size), mode='bilinear', align_corners=True)
    output_size = int(config.PRINTABLE_MIDDLE_STICKER_SIZE / 90 * sticker_size[2])
    middle_sticker = F.interpolate(middle_sticker, size=(output_size, output_size), mode='bilinear', align_corners=True)
    middle_shape_fake = F.interpolate(middle_shape_fake, size=(output_size, output_size), mode='bilinear', align_corners=True)
    # clipping stickers to fit shapes
    minimal = config.MINIMAL
    clipped_left_sticker = left_sticker.masked_fill(left_shape_fake < minimal, 1.0)
    clipped_right_sticker = right_sticker.masked_fill(right_shape_fake < minimal, 1.0)
    clipped_middle_sticker = middle_sticker.masked_fill(middle_shape_fake < minimal, 1.0)
    return clipped_left_sticker, clipped_right_sticker, clipped_middle_sticker


def random_location_sticker_and_shape(sticker, shape, sticker_setting, sticker_size, mode):
    '''
    random affine transformation to position stickers
    '''
    img_size = config.IMAGE_SIZE * 2
    half_img_size = img_size // 2
    batch_size, c, h, w = sticker.shape
    diff_h, diff_w = (img_size - h) // 2, (img_size - w) // 2
    padded_sticker = F.pad(sticker, pad=[diff_h, diff_h, diff_w, diff_w], mode='constant', value=0)
    padded_shape = F.pad(shape, pad=[diff_h, diff_h, diff_w, diff_w], mode='constant', value=0)

    # rotation
    angle = torch.FloatTensor(batch_size).uniform_(-10, 10) * math.pi / 180
    theta_rotation = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    '''
        [math.cos(angle), math.sin(-angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0]
    '''
    if mode == 'train':
        theta_rotation[:, 0, 0] = torch.cos(angle)
        theta_rotation[:, 0, 1] = torch.sin(-angle)
        theta_rotation[:, 1, 0] = torch.sin(angle)
        theta_rotation[:, 1, 1] = torch.cos(angle)

    # scaling
    scale = torch.FloatTensor(batch_size).uniform_(0.90, 1.10)
    theta_scaling = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    '''
        [scale, 0.0, 0.0],
        [0.0, scale, 0.0],
        [0.0, 0.0, 1.0]
    '''
    if mode == 'train':
        theta_scaling[:, 0, 0] = scale
        theta_scaling[:, 1, 1] = scale

    # translation
    translation_x = torch.FloatTensor(batch_size).uniform_(-10 / img_size, 10 / img_size)  # 10 / 300
    translation_y = torch.FloatTensor(batch_size).uniform_(-10 / img_size, 10 / img_size)  # 10 / 300
    theta_translation = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    '''
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
        [0.0, 0.0, 1.0]
    '''
    if mode == 'train':
        theta_translation[:, 0, 2] = 1 - sticker_setting[:, 0].cpu() / half_img_size + translation_x
        theta_translation[:, 1, 2] = 1 - (sticker_setting[:, 1].cpu() - (sticker_size // 2)) / half_img_size + translation_y
    else:
        theta_translation[:, 0, 2] = 1 - sticker_setting[:, 0].cpu() / half_img_size
        theta_translation[:, 1, 2] = 1 - (sticker_setting[:, 1].cpu() - (sticker_size // 2)) / half_img_size

    theta = torch.bmm(torch.bmm(theta_rotation, theta_scaling), theta_translation)[:, :2, :].to(padded_sticker.device)
    grid = F.affine_grid(theta, size=padded_sticker.size(), align_corners=True)
    projected_sticker = F.grid_sample(padded_sticker, grid, mode='nearest', align_corners=True)
    projected_shape = F.grid_sample(padded_shape, grid, mode='nearest', align_corners=True)
    return projected_sticker, projected_shape


def render_img(texture_sticker, texture_shape, config_sticker, image_size=config.IMAGE_SIZE):
    '''
    ref: https://github.com/facebookresearch/pytorch3d/issues/184
    The rendering function (just for test)
    Input:
        face_shape:  Tensor[1, 35709, 3]
        facemodel: contains `tri` (triangles[70789, 3], index start from 1)
    '''
    batch_size = texture_sticker.shape[0]
    device = texture_sticker.device
    face_shape, tri, gamma, intrinsic, uv_coords = config_sticker
    fx, px, fy, py = intrinsic[:, 0, 0].view(batch_size, -1), intrinsic[:, 0, 1].view(batch_size, -1), intrinsic[:, 0, 2].view(batch_size, -1), intrinsic[:, 0, 3].view(batch_size, -1)

    face_idx = tri.long() - 1 # index start from 1

    R = torch.eye(3).unsqueeze(0).repeat((batch_size, 1, 1)).to(device)
    R[:, 0, 0] *= -1.0
    T = torch.zeros([batch_size, 3]).to(device)

    half_size = (image_size - 1.0) / 2
    focal_length = torch.cat([fx / half_size, fy / half_size], dim=1)
    principal_point = torch.cat([(half_size - px) / half_size, (py - half_size) / half_size], dim=1)

    cameras = SfMPerspectiveCameras(
        device=device,
        R=R,
        T=T,
        focal_length=focal_length,
        principal_point=principal_point
    )

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1
    )

    blend_params = BlendParams(background_color=(.0, .0, .0))

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=TexturedSoftPhongShader(
            device=device,
            gamma=gamma,
            blend_params=blend_params
        )
    )

    # render sticker
    # materials = Materials(
    #     device=device,
    #     specular_color=[[1.0, 1.0, 1.0]],
    #     shininess=10.0
    # )
    # lights = PointLights(device=device, location=[[0.0, 0.0, 10.0]])

    texture_sticker = texture_sticker.permute((0, 2, 3, 1))
    face_color = TexturesUV(
        maps=texture_sticker,
        faces_uvs=face_idx,
        verts_uvs=uv_coords
    )
    mesh = Meshes(face_shape, face_idx, face_color)
    # removing specular light
    rendered_sticker = renderer(mesh)
    rendered_sticker = torch.clamp(rendered_sticker[:, :, :, :3], 0.0, 1.0)

    # render shape
    texture_shape = texture_shape.permute((0, 2, 3, 1))
    face_color = TexturesUV(
        maps=texture_shape,
        faces_uvs=face_idx,
        verts_uvs=uv_coords
    )
    mesh = Meshes(face_shape, face_idx, face_color)
    rendered_shape = renderer(mesh)
    rendered_shape = torch.clamp(rendered_shape[:, :, :, :3], 0.0, 1.0)

    return rendered_sticker.permute((0, 3, 1, 2)), rendered_shape.permute((0, 3, 1, 2))


def attach_stickers_to_image(left_sticker, left_shape, right_sticker, right_shape, middle_sticker, middle_shape, image, sticker_config, sticker_size, mode='train'):
    # convert [-1, 1] to [0, 1]
    left_sticker = convert_bound(left_sticker, 3)
    left_shape = convert_bound(left_shape, 3)
    right_sticker = convert_bound(right_sticker, 3)
    right_shape = convert_bound(right_shape, 3)
    middle_sticker = convert_bound(middle_sticker, 3)
    middle_shape = convert_bound(middle_shape, 3)
    image = convert_bound(image, 3)
    # size rescaling
    left_sticker = F.interpolate(left_sticker, size=(sticker_size[0], sticker_size[0]), mode='bilinear', align_corners=True)
    left_shape = F.interpolate(left_shape, size=(sticker_size[0], sticker_size[0]), mode='bilinear', align_corners=True)
    right_sticker = F.interpolate(right_sticker, size=(sticker_size[1], sticker_size[1]), mode='bilinear', align_corners=True)
    right_shape = F.interpolate(right_shape, size=(sticker_size[1], sticker_size[1]), mode='bilinear', align_corners=True)
    middle_sticker = F.interpolate(middle_sticker, size=(sticker_size[2], sticker_size[2]), mode='bilinear', align_corners=True)
    middle_shape = F.interpolate(middle_shape, size=(sticker_size[2], sticker_size[2]), mode='bilinear', align_corners=True)

    # random location
    locs = sticker_config[-1]
    # first, left sticker
    randomed_left_sticker, randomed_left_shape = random_location_sticker_and_shape(left_sticker, left_shape, locs[:, 0, :], sticker_size[0], mode)
    # second, right sticker
    randomed_right_sticker, randomed_right_shape = random_location_sticker_and_shape(right_sticker, right_shape, locs[:, 1, :], sticker_size[1], mode)
    # third, middle sticker
    randomed_middle_sticker, randomed_middle_shape = random_location_sticker_and_shape(middle_sticker, middle_shape, locs[:, 2, :], sticker_size[2], mode)

    randomed_left_shape = randomed_left_shape.repeat(1, 3, 1, 1)
    randomed_right_shape = randomed_right_shape.repeat(1, 3, 1, 1)
    randomed_middle_shape = randomed_middle_shape.repeat(1, 3, 1, 1)

    # render image
    # first, left sticker
    projected_left_sticker, projected_left_shape = render_img(randomed_left_sticker, randomed_left_shape, sticker_config[:-1])
    # second, right sticker
    projected_right_sticker, projected_right_shape = render_img(randomed_right_sticker, randomed_right_shape, sticker_config[:-1])
    # third, middle sticker
    projected_middle_sticker, projected_middle_shape = render_img(randomed_middle_sticker, randomed_middle_shape, sticker_config[:-1])

    MINIMAL = config.MINIMAL
    clipped_left_sticker = projected_left_sticker.masked_fill(projected_left_shape < MINIMAL, 0.0)
    clipped_right_sticker = projected_right_sticker.masked_fill(projected_right_shape < MINIMAL, 0.0)
    clipped_middle_sticker = projected_middle_sticker.masked_fill(projected_middle_shape < MINIMAL, 0.0)

    clipped_sticker = torch.clamp(clipped_left_sticker + clipped_right_sticker + clipped_middle_sticker, 0.0, 1.0)
    mask = torch.clamp(projected_left_shape + projected_right_shape + projected_middle_shape, 0.0, 1.0)

    image_with_sticker = image.masked_fill(mask > MINIMAL, 0.0) + clipped_sticker
    return mask, clipped_sticker, image_with_sticker
