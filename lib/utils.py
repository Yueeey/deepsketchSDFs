#!/usr/bin/env python3
import os
import pdb
import logging

import torch
import trimesh
import glob
import lib.workspace as ws
import numpy as np
import imageio

import pickle
# pytorch3d differentiable renderer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
)
# # Our own wrapper, to render contours
from lib.renderer_pytorch3D import ContourRenderer
# Image manipulation
from scipy.ndimage import rotate as rotate_scp
from scipy.ndimage.morphology import binary_dilation

class SoftThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold = 0.45, factor = 10.0):
        with torch.enable_grad():
            output = torch.sigmoid(factor*(input-threshold))
            ctx.save_for_backward(input, output)
        # binary thresholding
        return (input>threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        input.retain_grad()
        output.backward(grad_output, retain_graph=True)
        return input.grad

def compute_normal_consistency(gt_normal, pred_normal):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """
    gt_normal_np = gt_normal.float().detach().cpu().numpy()[0]
    gt_mask_np = (gt_normal[...,0]>0).float().detach().cpu().numpy()[0]

    pred_normal_np = pred_normal.float().detach().cpu().numpy()[0]
    pred_mask_np = (pred_normal[...,0]>0).float().detach().cpu().numpy()[0]

    # take valid intersection
    inner_mask = (gt_mask_np * pred_mask_np).astype(bool)

    gt_vecs = 2*gt_normal_np[inner_mask]-1
    pred_vecs = 2*pred_normal_np[inner_mask]-1
    metric = np.mean(np.sum(gt_vecs*pred_vecs, 1))

    return metric

class Renderer(torch.nn.Module):
    def __init__(self, silhouette_renderer, depth_renderer, max_depth = 5, image_size=256):
        super().__init__()
        self.silhouette_renderer = silhouette_renderer
        self.depth_renderer = depth_renderer

        self.max_depth = max_depth
        self.threshold = SoftThreshold()

        # sobel filters
        with torch.no_grad():
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cpu")

        ## INIT SOBEL FILTER
            k_filter = 3
            filter = self.get_sobel_kernel(k_filter)

            self.filter_x = torch.nn.Conv2d(in_channels=1,
                                            out_channels=1,
                                            kernel_size=k_filter,
                                            padding=0,
                                            bias=False)
            self.filter_x.weight[:] = torch.tensor(filter, requires_grad = False)
            self.filter_x = self.filter_x.to(self.device)

            self.filter_y = torch.nn.Conv2d(in_channels=1,
                                            out_channels=1,
                                            kernel_size=k_filter,
                                            padding=0,
                                            bias=False)
            self.filter_y.weight[:] = torch.tensor(filter.T, requires_grad = False)
            self.filter_y = self.filter_y.to(self.device)

        # Pixel coordinates
        self.X, self.Y = torch.meshgrid(torch.arange(0, image_size), torch.arange(0, image_size))
        self.X = (2*(0.5 + self.X.unsqueeze(0).unsqueeze(-1))/image_size - 1).float().cuda()
        self.Y = (2*(0.5 + self.Y.unsqueeze(0).unsqueeze(-1))/image_size - 1).float().cuda()

    def get_sobel_kernel(self, k=3):
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        return sobel_2D
    
    def depth_2_normal(self, depth, depth_unvalid, cameras):

        B, H, W, C = depth.shape

        grad_out = torch.zeros(B, H, W, 3).cuda()
        # Pixel coordinates
        xy_depth = torch.cat([self.X, self.Y, depth], 3).cuda().reshape(B,-1, 3)
        xyz_unproj = cameras.unproject_points(xy_depth, world_coordinates=False)

        # compute tangent vectors
        XYZ_camera = xyz_unproj.reshape(B, H, W, 3)
        vx = XYZ_camera[:,1:-1,2:,:]-XYZ_camera[:,1:-1,1:-1,:]
        vy = XYZ_camera[:,2:,1:-1,:]-XYZ_camera[:,1:-1,1:-1,:]

        # finally compute cross product
        normal = torch.cross(vx.reshape(-1, 3),vy.reshape(-1, 3))
        normal_norm = normal.norm(p=2, dim=1, keepdim=True)

        normal_normalized = normal.div(normal_norm)
        # reshape to image
        normal_out = normal_normalized.reshape(B, H-2, W-2, 3)
        grad_out[:,1:-1,1:-1,:] = (0.5 - 0.5*normal_out)

        # zero out +Inf
        grad_out[depth_unvalid] = 0.0

        return grad_out

    def buffer_2_contour(self, buffer):
        # set the steps tensors
        B, C, H, W = buffer.shape
        grad = torch.zeros((B, 1, H, W)).to(self.device)
        padded_buffer = torch.nn.functional.pad(buffer, (1,1,1,1), mode='reflect')
        for c in range(C):
            grad_x = self.filter_x(padded_buffer[:, c:c+1])
            grad_y = self.filter_y(padded_buffer[:, c:c+1])
            grad_tensor = torch.stack((grad_x, grad_y),-1)
            grad_magnitude = torch.norm(grad_tensor, p =2, dim = -1)
            grad = grad + grad_magnitude

        return self.threshold.apply(1.0 - (torch.clamp(grad,0,1)))

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        # take care of soft silhouette
        silhouette_ref = self.silhouette_renderer(meshes_world=meshes_world, **kwargs)
        silhouette_out = silhouette_ref[..., 3]

        # now get depth out
        depth_ref = self.depth_renderer(meshes_world=meshes_world, **kwargs)
        depth_ref = depth_ref.zbuf[...,0].unsqueeze(-1)
        depth_unvalid = depth_ref<0
        depth_ref[depth_unvalid] = self.max_depth
        depth_out = depth_ref[..., 0]

        # post process depth to get normals, contours
        normals_out = self.depth_2_normal(depth_ref, depth_unvalid.squeeze(-1), kwargs['cameras'])

        return normals_out, silhouette_out

    def contour(self, meshes_world, **kwargs) -> torch.Tensor:
        # take care of soft silhouette
        silhouette_ref = self.silhouette_renderer(meshes_world=meshes_world, **kwargs)
        silhouette_out = silhouette_ref[..., 3]

        # now get depth out
        depth_ref = self.depth_renderer(meshes_world=meshes_world, **kwargs)
        depth_ref = depth_ref.zbuf[...,0].unsqueeze(-1)
        depth_unvalid = depth_ref<0
        depth_ref[depth_unvalid] = self.max_depth
        depth_out = depth_ref[..., 0]

        # post process depth to get normals, contours
        normals_out = self.depth_2_normal(depth_ref, depth_unvalid.squeeze(-1), kwargs['cameras']).permute(0,3,1,2)
        contours_out = self.buffer_2_contour(
                                torch.cat(( normals_out,
                                            depth_ref.permute(0,3,1,2))
                                    , 1)
                                )

        return contours_out

def process_image(images_out, alpha_out):
    image_out_export = 255*images_out.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
    alpha_out_export = 255*alpha_out.detach().cpu().numpy()[0]
    image_out_export = np.concatenate( (image_out_export, alpha_out_export[:,:,np.newaxis]), -1 )
    return image_out_export.astype(np.uint8)

def store_image(image_filename, images_out, alpha_out):
    image_out_export = process_image(images_out, alpha_out)
    imageio.imwrite(image_filename, image_out_export)

def interpolate_on_faces(field, faces):
    #TODO: no batch support for now
    nv = field.shape[0]
    nf = faces.shape[0]
    field = field.reshape((nv, 1))
    # pytorch only supports long and byte tensors for indexing
    face_coordinates = field[faces.long()].squeeze(0)
    centroids = 1.0/3 * torch.sum(face_coordinates, 1)
    return centroids

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_logs(
    experiment_directory,
    loss_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["epoch"],
    )


def clip_logs(loss_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)
    loss_log = loss_log[: (iters_per_epoch * epoch)]

    return loss_log


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def fourier_transform(x, L=5):
    cosines = torch.cat([torch.cos(2**l*3.1415*x) for l in range(L)], -1)
    sines = torch.cat([torch.sin(2**l*3.1415*x) for l in range(L)], -1)
    transformed_x = torch.cat((cosines,sines),-1)
    return transformed_x


def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]
    latent_repeat = latent_vector.expand(num_samples, -1)
    sdf = decoder(latent_repeat, queries)
    return sdf

def get_instance_filenames(data_source, split):
    npzfiles = []
    split_file = os.path.join(data_source, str(split) + '.lst')
    with open(split_file, 'r') as f:
            models_c = f.read().split('\n')
    models_c = list(filter(lambda x: len(x) > 0, models_c))
    for instance_name in models_c:
        instance_filename = os.path.join(
                    data_source, 'chairs', 'samples', instance_name + ".npz"
                )
        if not os.path.isfile(instance_filename):
            logging.warning(
                "Requested non-existent file '{}'".format(instance_filename)
            )
        npzfiles += [instance_filename]
    return npzfiles


    # for dataset in split:
    #     for class_name in split[dataset]:
    #         for instance_name in split[dataset][class_name]:
    #             instance_filename = os.path.join(
    #                 dataset, class_name, instance_name + ".npz"
    #             )
    #             if not os.path.isfile(
    #                 os.path.join(data_source, instance_filename)
    #             ):
    #                 logging.warning(
    #                     "Requested non-existent file '{}'".format(instance_filename)
    #                 )
    #             npzfiles += [instance_filename]
    # return npzfiles


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def unpack_sdf_samples(filename, subsample=None):

    npz = np.load(filename, allow_pickle=True)
    # if subsample is None:
    #     return npz

    pos_tensor = remove_nans(torch.from_numpy(npz["pos"].astype(float)))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"].astype(float)))
    
    if subsample is None:
        samples = torch.cat([pos_tensor, pos_tensor], 0).float()
        return samples

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half).cpu() * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half).cpu() * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0).float()

    return samples

def read_params(lines):
    params = []
    for line in lines:
        line = line.strip()[1:-2]
        param = np.fromstring(line, dtype=float, sep=',')
        params.append(param)
    return params


def get_rotate_matrix(rotation_angle1):
    cosval = np.cos(rotation_angle1)
    sinval = np.sin(rotation_angle1)

    rotation_matrix_x = np.array([[1, 0, 0, 0],
                                  [0, cosval, -sinval, 0],
                                  [0, sinval, cosval, 0],
                                  [0, 0, 0, 1]])
    rotation_matrix_y = np.array([[cosval, 0, sinval, 0],
                                  [0, 1, 0, 0],
                                  [-sinval, 0, cosval, 0],
                                  [0, 0, 0, 1]])
    rotation_matrix_z = np.array([[cosval, -sinval, 0, 0],
                                  [sinval, cosval, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    scale_y_neg = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    neg = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    return np.linalg.multi_dot([neg, rotation_matrix_z, rotation_matrix_z, scale_y_neg, rotation_matrix_x])

rot90y = np.array([[0, 0, -1],
                   [0, 1, 0],
                   [1, 0, 0]], dtype=np.float32)

def getBlenderProj(az, el, distance_ratio, roll = 0, focal_length=35, img_w=137, img_h=137):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""
    F_MM = focal_length  # Focal length
    SENSOR_SIZE_MM = 32.
    PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
    RESOLUTION_PCT = 100.
    SKEW = 0.
    CAM_MAX_DIST = 1.75
    CAM_ROT = np.asarray([[1.910685676922942e-15, 4.371138828673793e-08, 1.0],
                      [1.0, -4.371138828673793e-08, -0.0],
                      [4.371138828673793e-08, 1.0, -4.371138828673793e-08]])

    # Calculate intrinsic matrix.
    scale = RESOLUTION_PCT / 100
    # print('scale', scale)
    f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
    f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
    # print('f_u', f_u, 'f_v', f_v)
    u_0 = img_w * scale / 2
    v_0 = img_h * scale / 2
    K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    # Step 1: World coordinate to object coordinate.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_world2obj = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
                                          (sa * ce, ca, sa * se),
                                          (-se, 0, ce))))

    # Step 2: Object coordinate to camera coordinate.
    R_obj2cam = np.transpose(np.matrix(CAM_ROT))
    R_world2cam = R_obj2cam * R_world2obj
    cam_location = np.transpose(np.matrix((distance_ratio * CAM_MAX_DIST,
                                           0,
                                           0)))
    # print('distance', distance_ratio * CAM_MAX_DIST)
    T_world2cam = -1 * R_obj2cam * cam_location

    # Step 3: Fix blender camera's y and z axis direction.
    R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_world2cam = R_camfix * R_world2cam
    T_world2cam = R_camfix * T_world2cam

    RT = np.hstack((R_world2cam, T_world2cam))
    # finally, consider roll
    cr = np.cos(np.radians(roll))
    sr = np.sin(np.radians(roll))
    R_z = np.matrix(((cr, -sr, 0),
                  (sr, cr, 0),
                  (0, 0, 1)))
    return K, R_z@RT


def get_camera_matrices(metadata_filename, id):
    # Adaptation of Code/Utils from DISN
    camera_dict = np.load(metadata_filename)
    # camera_dict = np.load(camera_file)
    Rt = camera_dict['world_mat_%d' % id].astype(np.float32)
    K = camera_dict['camera_mat_%d' % id].astype(np.float32)
    roll = 0
    cr = np.cos(np.radians(roll))
    sr = np.sin(np.radians(roll))
    R_z = np.matrix(((cr, -sr, 0),
                  (sr, cr, 0),
                  (0, 0, 1)))
    RT = R_z@Rt
    rot_mat = get_rotate_matrix(-np.pi / 2)
    extrinsic = np.linalg.multi_dot([RT, rot_mat])

    intrinsic = torch.tensor(K).float()
    extrinsic = torch.tensor(extrinsic).float()

    # with open(metadata_filename, 'r') as f:
    #     lines = f.read().splitlines()
    #     param_lst = read_params(lines)
    #     rot_mat = get_rotate_matrix(-np.pi / 2)
    #     az, el, distance_ratio = param_lst[id][0], param_lst[id][1], param_lst[id][3]
    #     intrinsic, RT = getBlenderProj(az, el, distance_ratio, img_w=224, img_h=224)
    #     extrinsic = np.linalg.multi_dot([RT, rot_mat])
    # intrinsic = torch.tensor(intrinsic).float()
    # extrinsic = torch.tensor(extrinsic).float()

    return intrinsic, extrinsic

def get_projection(az, el, distance, focal_length=35, img_w=256, img_h=256, sensor_size_mm = 32.):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""

    # Calculate intrinsic matrix.
    f_u = focal_length * img_w  / sensor_size_mm
    f_v = focal_length * img_h  / sensor_size_mm
    u_0 = img_w / 2
    v_0 = img_h / 2
    K = np.matrix(((f_u, 0, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    sa = np.sin(np.radians(az))
    ca = np.cos(np.radians(az))
    R_azimuth = np.transpose(np.matrix(((ca, 0, sa),
                                          (0, 1, 0),
                                          (-sa, 0, ca))))
    se = np.sin(np.radians(el))
    ce = np.cos(np.radians(el))
    R_elevation = np.transpose(np.matrix(((1, 0, 0),
                                          (0, ce, -se),
                                          (0, se, ce))))
    # fix up camera
    se = np.sin(np.radians(180))
    ce = np.cos(np.radians(180))
    R_cam = np.transpose(np.matrix(((ce, -se, 0),
                                          (se, ce, 0),
                                          (0, 0, 1))))
    T_world2cam = np.transpose(np.matrix((0,
                                           0,
                                           distance)))
    RT = np.hstack((R_cam@R_elevation@R_azimuth, T_world2cam))

    return K, RT

def unpack_images(filename):

    image = imageio.imread(filename).astype(float)/255.0
    return torch.tensor(image).float().permute(2,0,1)

###########From sketch2mesh

def get_projection_torch3D(az, el, distance, focal_length=35, img_w=256, img_h=256, sensor_size_mm = 32., RCAM=False):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""
    # Calculate intrinsic matrix.
    K = np.eye(4)
    K[0][0] = 2.1875
    K[1][1] = 2.1875
    K = K.astype(np.float32)

    # Calculate rotation and translation matrices.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    # Edo's convention
    #sa = np.sin(np.radians(az+90))
    #ca = np.cos(np.radians(az+90))
    R_azimuth = np.transpose(np.matrix(((ca, 0, sa),
                                          (0, 1, 0),
                                          (-sa, 0, ca))))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_elevation = np.transpose(np.matrix(((1, 0, 0),
                                          (0, ce, -se),
                                          (0, se, ce))))
    # fix up camera
    se = np.sin(np.radians(90))
    ce = np.cos(np.radians(90))
    if RCAM:
        R_cam = np.transpose(np.matrix(((ce, -se, 0),
                                            (se, ce, 0),
                                            (0, 0, 1))))
    else:
        R_cam = np.transpose(np.matrix(((1, 0, 0),
                                        (0, 1, 0),
                                        (0, 0, 1))))
    T_world2cam = np.transpose(np.matrix((0,
                                           0,
                                           distance)))
    RT = np.hstack((R_cam@R_azimuth@R_elevation, T_world2cam))

    return RT, K

def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


class AverageValueMeter(object):
    """
    Computes and stores the average and current value of a sequence of floats
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.min = min(self.min, val)
        self.max = max(self.max, val)


class ObjectMetricTracker():
    """
    Store metrics for one object along the course of 1 refinement
    """
    def __init__(self, metrics=['chd']):
        self.metrics = {}
        self.steps = {}
        self.best_metrics = {}
        for m in metrics:
            self.metrics[m] = []
            self.steps[m] = []
            self.best_metrics[m] = 1000000.

    def append(self, m, value, step):
        """
        Stores the latest value of metric m,
        returns true if a minimum is reached,
        false otherwise
        """
        if not m in self.metrics:
            self.metrics[m] = []
            self.steps[m] = []
            self.best_metrics[m] = 1000000.
        self.metrics[m].append(value)
        self.steps[m].append(step)
        if self.metrics[m][-1] < self.best_metrics[m]:
            self.best_metrics[m] = self.metrics[m][-1]
            return True
        return False
        
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))


def myChamferDistance(x, y):  # for example, x = 2025,3 y = 2048,3
    #   compute chamfer distance between two point clouds x and y
    x_size = x.size()
    y_size = y.size()
    assert (x_size[1] == y_size[1]) # Same dimensionality of pts
    x = torch.unsqueeze(x, 0)  # x = 1,2025,3
    y = torch.unsqueeze(y, 1)  # y = 2048,1,3
    x = x.repeat(y_size[0], 1, 1)  # x = 2048,2025,3
    y = y.repeat(1, x_size[0], 1)  # y = 2048,2025,3
    x_y = x - y
    x_y = torch.pow(x_y, 2)  # x_y = 2048,2025,3
    x_y = torch.sum(x_y, 2, keepdim=True)  # x_y = 2048,2025,1
    x_y = torch.squeeze(x_y, 2)  # x_y = 2048,2025
    x_y_row, _ = torch.min(x_y, 0, keepdim=True)  # x_y_row = 1,2025
    x_y_col, _ = torch.min(x_y, 1, keepdim=True)  # x_y_col = 2048,1
    return x_y_row, x_y_col.squeeze(-1).unsqueeze(0)


def contours_pointcloud(verts, faces, contours, instanciated_renderer, cam):
    """
    Input:
        verts: [_, 3]
        faces: [_, 3]
        contours: [1, 1, 256, 256]
        instanciated_renderer: a Renderer, with .depth_renderer (MeshRasterizer) attribute
        cam: FoVPerspectiveCameras
    Returns: 2d coordinates of contour points, in the range [-1,1]^2
        Shape: Nx2

    TODO: this does not support batching yet... issue is at line
        f_inds = pix_to_face[contours[0] < 0.5]
    TODO: make code more efficient, since projection to screen space is 3 times in total
        (once previously to render contours, once manually here, once in the renderer to get the fragment)
    """
    # Pack verts+faces in a mesh structure:
    meshes = Meshes(verts.unsqueeze(0), faces.unsqueeze(0))
    # Project it to screen space
    meshes_screen = instanciated_renderer.depth_renderer.transform(meshes, cameras=cam)
    # Get the vertices coordinates of projected faces
    proj_faces = meshes_screen.verts_packed()[meshes_screen.faces_packed()] # (N_faces, 3, 3)

    # Render a fragment, for each pixel getting the face id and barycentric coords
    #with torch.no_grad():
    fragment = instanciated_renderer.depth_renderer(meshes, cameras=cam)
    pix_to_face, bary_coords = fragment.pix_to_face, fragment.bary_coords # long (batch, H, W, 1) and float (batch, H, W, 1, 3])

    # Keep only points from contours
    f_inds = pix_to_face[contours[0] < 0.5]     # index of all faces projected to contours (K, 1)
    weights = bary_coords[contours[0] < 0.5]    # barycentric coordinates (K, 1, 3)
    # Filter out points that fall outside the mesh (consequence of Sobel finite diff.: slightly bleeding contours)
    weights = weights[f_inds > 0]   # (L, 3), with L <= K
    f_inds = f_inds[f_inds>0]       # (L)
    # Coordinates of the points: perform barycentric interpolation
    pts = torch.bmm(weights.unsqueeze(1), proj_faces[f_inds]) # (L,1,3)
    pts = pts.squeeze(1)    # (L,3) - u,v,depth

    # Put in 2D uv coordinates
    coords_2d_uv = pts[:,:2]                        # remove depth component
    coords_2d_uv = torch.flip(coords_2d_uv, [1])    # swap x<->y axis
    coords_2d_uv = -coords_2d_uv        # flip both axis

    return coords_2d_uv


def pack_mesh_and_render(verts, faces, instanciated_renderer, cam, light):
    # Pack mesh and create fake texture
    meshes = Meshes(verts.unsqueeze(0), faces.unsqueeze(0))
    verts_shape = meshes.verts_packed().shape
    sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=verts.device, requires_grad=False)
    meshes.textures = TexturesVertex(verts_features=sphere_verts_rgb)
    # meshes = Meshes(verts_dr, faces_dr)
    # verts_shape = meshes.verts_packed().shape
    # verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=False)
    # meshes.textures = TexturesVertex(verts_features=verts_rgb)

    # Render
    return instanciated_renderer.contour(meshes, cameras=cam, lights=light)


def get_renderer_cameras_lights(cameras, lights, image_size=256):
    # device = R_cuda.device
    with torch.no_grad():
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
    # lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    # cameras = FoVPerspectiveCameras(device=device, znear=0.001, zfar=3500, aspect_ratio=1.0, fov=60.0, R=R_cuda, T=t_cuda)
        
    sigma = 1e-5
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.000001, # To avoid bumps in the depth map > grainy sketches
        faces_per_pixel=1,
    )
    raster_settings_soft = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
        faces_per_pixel=25,
    )
    # silhouette renderer
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings_soft
        ),
        shader=SoftSilhouetteShader()
    )
    # depth renderer
    depth_renderer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    # image renderer
    image_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )
    # assemble in single rendering function
    renderer_pytorch3D = ContourRenderer(silhouette_renderer, depth_renderer, image_renderer)

    return renderer_pytorch3D


def find_min_max(contours, coords, imsize):
    ### Find min/max along x direction
    # Min x
    idx_0 = np.argmin(contours, axis=0)
    idx_0 = np.stack([idx_0, coords], axis=1)[idx_0 > 0]
    # Max x
    idx_1 = np.argmin(contours[::-1,:], axis=0)
    idx_1 = np.stack([idx_1, coords], axis=1)[idx_1 > 0]
    idx_1[:,0] = imsize - idx_1[:,0] - 1
    ### Find min/max along y direction
    # Min y
    idy_0 = np.argmin(contours, axis=1)
    idy_0 = np.stack([coords, idy_0], axis=1)[idy_0 > 0]
    # Max y
    idy_1 = np.argmin(contours[:,::-1], axis=1)
    idy_1 = np.stack([coords, idy_1], axis=1)[idy_1 > 0]
    idy_1[:,1] = imsize - idy_1[:,1] - 1
    ### Create output image
    img_pc = np.ones((imsize, imsize))
    idxy = np.concatenate((idx_0, idx_1, idy_0, idy_1))
    img_pc[idxy[:,0], idxy[:,1]] = 0.
    return img_pc


def filter_contours_exterior(contours, dilation=False,
        degs = [10, 20, 30, 35, 40, 45]):
    """
    Filter a contour (binary) image, to keep only the outmost pixels
    and remove what can be seen as 'interior' pixels - without assuming
    or needing that the outmost contour is closed

    Input:
        contours: shape (N*N)
        dilation: boolean, do we thicken the returned outmost contour? 
            Can be useful after Sobel filters, that gives slightly bleeding contours,
            to be alined in 3D space
        degs: in addition to rays perpendicular to the image borders,
            what are the orientations we shoot at?
    Output:
        shape (N*N)
    """
    contours = contours.squeeze()
    imsize = contours.shape[0]
    coords = np.arange(0, imsize)
    img_pc = find_min_max(contours, coords, imsize)
    for orientation in [-1, 1]:
        for d in degs:
            rotated_im = rotate_scp(contours, d * orientation, order=0, reshape=False, cval=1.)
            filtered_rotated_im = find_min_max(rotated_im, coords, imsize)
            img_pc = np.minimum(img_pc, rotate_scp(filtered_rotated_im, - d * orientation, order=0, reshape=False, cval=1.))
    # Dilate if needed:
    if dilation:
        img_pc = 1 - binary_dilation(1-img_pc, iterations=2)
    # Take intersection with original image
    return 1 - (1 - img_pc) * (1 - contours)


def filter_contours_exterior_thick(contours, dilation=True,
    degs = [0]
        ):
    """
    Filter a contour (binary) image, to keep only the outmost pixels
    and remove what can be seen as 'interior' pixels - without assuming
    or needing that the outmost contour is closed
    Input:
        contours: shape (N*N)
        dilation: boolean, do we thicken the returned outmost contour?
            Can be useful after Sobel filters, that gives slightly bleeding contours,
            to be alined in 3D space
        degs: in addition to rays perpendicular to the image borders,
            what are the orientations we shoot at?
    Output:
        shape (N*N)
    """
    imsize = contours.shape[0]
    coords = np.arange(0, imsize)
    img_pc = find_min_max_np_inner(contours, coords, imsize)
    for orientation in [-1, 1]:
        for d in degs:
            rotated_im = rotate_scp(contours, d * orientation, order=0, reshape=False, cval=1.)
            filtered_rotated_im = find_min_max_np_inner(rotated_im, coords, imsize)
            img_pc = np.minimum(img_pc, rotate_scp(filtered_rotated_im, - d * orientation, order=0, reshape=False, cval=1.))
    # Dilate if needed:
    if dilation:
        img_pc = 1 - binary_dilation(1-img_pc, iterations=1)
    # Take intersection with original image
    return 1 - (1 - img_pc) * (1 - contours)



def are_contours_thick(contours):
    ### Find min/max along x direction
    # Min x
    idx_0 = np.argmin(contours, axis=0)
    contours_2 = contours.copy()
    for i in range(len(idx_0)):
        contours_2[:idx_0[i], i] = 0
    idx_0_inner = np.argmax(contours_2, axis=0) - 1 - 2
    diff_a = idx_0_inner - idx_0
    # Max x
    idx_1 = np.argmin(contours[::-1,:], axis=0)
    contours_2 = contours[::-1,:].copy()
    for i in range(len(idx_1)):
        contours_2[:idx_1[i], i] = 0
    idx_1_inner = np.argmax(contours_2, axis=0) - 1 - 2
    diff_b = idx_1_inner - idx_1
    ### Find min/max along y direction
    # Min y
    idy_0 = np.argmin(contours, axis=1)
    contours_2 = contours.copy()
    for i in range(len(idy_0)):
        contours_2[i, :idy_0[i]] = 0
    idy_0_inner = np.argmax(contours_2, axis=1) - 1 - 2
    diff_c = idy_0_inner - idy_0
    # Max y
    idy_1 = np.argmin(contours[:,::-1], axis=1)
    contours_2 = contours[:,::-1].copy()
    for i in range(len(idy_1)):
        contours_2[i, :idy_1[i]] = 0
    idy_1_inner = np.argmax(contours_2, axis=1) - 1 - 2
    diff_d = idy_1_inner - idy_1
    # Heuristic on difference between inner/outer contours
    thick_a = diff_a[(diff_a > -3)*(diff_a < 6)].mean() > 1.2
    thick_b = diff_b[(diff_b > -3)*(diff_b < 6)].mean() > 1.2
    thick_c = diff_c[(diff_c > -3)*(diff_c < 6)].mean() > 1.2
    thick_d = diff_d[(diff_d > -3)*(diff_d < 6)].mean() > 1.2
    return thick_a and thick_b and thick_c and thick_d


# Inner
def find_min_max_np_inner(contours, coords, imsize):
    ### Find min/max along x direction
    # Min x
    idx_0 = np.argmin(contours, axis=0)
    # UGLY LOOP: fill contours up to idx_0
    contours_2 = contours.copy()
    for i in range(len(idx_0)):
        contours_2[:idx_0[i], i] = 0
    idx_0_inner = np.argmax(contours_2, axis=0) - 1 - 2
    idx_0 = np.stack([idx_0_inner, coords], axis=1)[idx_0 > 0]
    #
    #
    # Max x
    idx_1 = np.argmin(contours[::-1,:], axis=0)
    # UGLY LOOP: fill contours up to idx_1
    contours_2 = contours[::-1,:].copy()
    for i in range(len(idx_1)):
        contours_2[:idx_1[i], i] = 0
    idx_1_inner = np.argmax(contours_2, axis=0) - 1 - 2
    idx_1 = np.stack([idx_1_inner, coords], axis=1)[idx_1 > 0]
    idx_1[:,0] = imsize - idx_1[:,0] - 1
    #
    #
    ### Find min/max along y direction
    # Min y
    idy_0 = np.argmin(contours, axis=1)
    # UGLY LOOP: fill contours up to idy_0
    contours_2 = contours.copy()
    for i in range(len(idy_0)):
        contours_2[i, :idy_0[i]] = 0
    idy_0_inner = np.argmax(contours_2, axis=1) - 1 - 2
    idy_0 = np.stack([coords, idy_0_inner], axis=1)[idy_0 > 0]
    #
    #
    # Max y
    idy_1 = np.argmin(contours[:,::-1], axis=1)
    # UGLY LOOP: fill contours up to idy_1
    contours_2 = contours[:,::-1].copy()
    for i in range(len(idy_1)):
        contours_2[i, :idy_1[i]] = 0
    idy_1_inner = np.argmax(contours_2, axis=1) - 1 - 2
    idy_1 = np.stack([coords, idy_1_inner], axis=1)[idy_1 > 0]
    idy_1[:,1] = imsize - idy_1[:,1] - 1
    #
    #
    ### Create output image
    img_pc = np.ones((imsize, imsize))
    idxy = np.concatenate((idx_0, idx_1, idy_0, idy_1))
    img_pc[idxy[:,0], idxy[:,1]] = 0.
    return img_pc

def filter_contours_input(contours):
    if are_contours_thick(contours):
        return filter_contours_exterior_thick(contours)
    else:
        return filter_contours_exterior(contours)