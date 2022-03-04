import torch
import torch.nn as nn
from pytorch3d.renderer import BlendParams
from pytorch3d.renderer.mesh.utils import interpolate_face_attributes
from pytorch3d.renderer.blending import softmax_rgb_blend
from typing import Tuple
import numpy as np


class _need_const:
    a0 = np.pi
    a1 = 2 * np.pi / np.sqrt(3.0)
    a2 = 2 * np.pi / np.sqrt(8.0)
    c0 = 1 / np.sqrt(4 * np.pi)
    c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
    c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
    d0 = 0.5 / np.sqrt(3.0)

    illu_consts = [a0, a1, a2, c0, c1, c2, d0]


def _apply_lighting(
    points, normals, gamma, lights, cameras, materials
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    compute vertex color using face_texture and SH function lighting approximation
    Args:
        points: torch tensor of shape (N, P, 3) or (P, 3).
        normals: torch tensor of shape (N, P, 3 3) or (P, 3)
        gamma: torch tensor of Shape (N, 27) of (27)
    Returns:
        ambient_color: same shape as materials.ambient_color
        diffuse_color: same shape as the input points
        specular_color: same shape as the input points
    """
    n_b, n_points, n_points, k, _ = points.size()
    n_v_full = n_b * n_points * n_points * k
    gamma = gamma.view(-1, 3, 9).clone()
    gamma[:, :, 0] += 0.8

    gamma = gamma.permute(0, 2, 1)

    a0, a1, a2, c0, c1, c2, d0 = _need_const.illu_consts

    Y0 = torch.ones(n_v_full).float() * a0*c0
    if gamma.is_cuda:
        Y0 = Y0.cuda()
    norm = normals.view(-1, 3)
    nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
    arrH = []

    arrH.append(Y0)
    arrH.append(-a1*c1*ny)
    arrH.append(a1*c1*nz)
    arrH.append(-a1*c1*nx)
    arrH.append(a2*c2*nx*ny)
    arrH.append(-a2*c2*ny*nz)
    arrH.append(a2*c2*d0*(3*nz.pow(2)-1))
    arrH.append(-a2*c2*nx*nz)
    arrH.append(a2*c2*0.5*(nx.pow(2)-ny.pow(2)))

    H = torch.stack(arrH, 1)
    Y = H.view(n_b, n_points, n_points, k, 9)

    # Y shape:[batch, H, W, K, 9]
    # shape:[batch, 9, 3]
    lighting = torch.einsum('nhwki,nij->nhwkj', Y, gamma)

    if lights is None and cameras is None and materials is None:
        specular_color = 0
    else:
        # specular color
        light_specular = lights.specular(
            normals=normals,
            points=points,
            camera_position=cameras.get_camera_center(),
            shininess=materials.shininess,
        )
        specular_color = materials.specular_color * light_specular
    return lighting, specular_color


def phong_shading(
    meshes, fragments, gamma, lights, cameras, materials, texels
) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    vertex_normals[:, 0] *= -1
    vertex_normals[:, 1] *= -1
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    ambient, specular = _apply_lighting(
        pixel_coords, pixel_normals, gamma, lights, cameras, materials
    )
    colors = ambient * texels + specular
    return colors


class TexturedSoftPhongShader(nn.Module):
    """
    Per pixel lighting applied to a texture map. First interpolate the vertex
    uv coordinates and sample from a texture map. Then apply the lighting model
    using the interpolated coords and normals for each pixel.

    The blending function returns the soft aggregated color using all
    the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = TexturedPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cpu", gamma=None, blend_params=None
    ):
        super().__init__()
        self.lights = None
        self.materials = None
        self.cameras = None
        self.gamma = gamma
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get("blend_params", self.blend_params)
        cameras = kwargs.get("cameras", self.cameras)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            gamma=self.gamma,
            cameras=cameras,
            materials=materials,
            lights=lights
        )
        images = softmax_rgb_blend(colors, fragments, blend_params)
        return images
