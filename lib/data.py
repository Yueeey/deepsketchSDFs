#!/usr/bin/env python3

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import pdb
import imageio
import time
import random
import imageio

from lib.utils import *


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample
    ):
        self.subsample = subsample
        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)


    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        return unpack_sdf_samples(filename, self.subsample), idx, self.npyfiles[idx]


class RGBA2SDF(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)
        self.img_source = '/vol/research/ycau/shapeNet/03001627'

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        mesh_name = self.npyfiles[idx].split(".npz")[0].split('/')[-1]
        # fetch sdf samples
        sdf_filename = self.npyfiles[idx]
        sdf_samples = unpack_sdf_samples(sdf_filename, self.subsample)

        if self.is_train:
            # reset seed for random sampling training data (see https://github.com/pytorch/pytorch/issues/5059)
            np.random.seed( int(time.time()) + idx)
            id = np.random.randint(0, self.num_views)
        else:
            # np.random.seed(idx)
            # id = np.random.randint(0, self.num_views)

            id = 5

        view_id = '{0:02d}'.format(id)

        image_folder = os.path.join(self.data_source, 'chairs', 'renders', mesh_name, 'naive_mad', 'base')
        image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
        image_files.sort()
        image_filename = image_files[id]
        RGBA = unpack_images(image_filename)

        sil_folder = os.path.join(self.data_source, 'chairs', 'renders', mesh_name, 'sil_mad', 'base')
        sil_files = glob.glob(os.path.join(sil_folder, '*.png'))
        sil_files.sort()
        sil_filename = sil_files[id]
        sil = unpack_images(sil_filename)
        sil = 1 - sil
        sil = sil[0]

        azi_list = [0, 135, 180, 225, 270, 315, 45, 90]
        azi = azi_list[id]

        # fetch cameras
        # metadata_filename = os.path.join(self.img_source, mesh_name, 'camera', 'cameras.npz')
        # intrinsic, extrinsic = get_camera_matrices(metadata_filename, id)

        # return sdf_samples, RGBA, sil, intrinsic, azi, mesh_name
        return sdf_samples, RGBA, sil, azi, mesh_name

class RGBA2SDF_MULTI(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)
        self.img_source = '/vol/research/ycau/shapeNet/03001627'

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        mesh_name = self.npyfiles[idx].split(".npz")[0].split('/')[-1]
        # fetch sdf samples
        sdf_filename = self.npyfiles[idx]
        sdf_samples = unpack_sdf_samples(sdf_filename, self.subsample)
        
        RGBA_list = []
        sil_list = []
        intrinsic_list = []
        azi_list = []

        for i in range(self.num_views):
            id = i

            view_id = '{0:02d}'.format(id)

            image_folder = os.path.join(self.data_source, 'chairs', 'renders', mesh_name, 'naive_mad', 'base')
            image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
            image_files.sort()
            image_filename = image_files[id]
            RGBA = unpack_images(image_filename)
            RGBA_list.append(RGBA)

            sil_folder = os.path.join(self.data_source, 'chairs', 'renders', mesh_name, 'sil_mad', 'base')
            sil_files = glob.glob(os.path.join(sil_folder, '*.png'))
            sil_files.sort()
            sil_filename = sil_files[id]
            sil = unpack_images(sil_filename)
            sil = 1 - sil
            sil = sil[0]
            sil_list.append(sil)

            azi_lists = [0, 135, 180, 225, 270, 315, 45, 90]
            azi = azi_lists[id]
            azi_list.append(azi)

            # fetch cameras
            metadata_filename = os.path.join(self.img_source, mesh_name, 'camera', 'cameras.npz')
            intrinsic, extrinsic = get_camera_matrices(metadata_filename, id)
            intrinsic_list.append(intrinsic)

        return sdf_samples, RGBA_list, sil_list, intrinsic_list, azi_list, mesh_name


class RGBA2SDF_TEST(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)
        self.img_source = '/vol/research/ycau/shapeNet/03001627'

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        mesh_name = self.npyfiles[idx].split(".npz")[0].split('/')[-1]
        # fetch sdf samples
        sdf_filename = self.npyfiles[idx]
        sdf_samples = unpack_sdf_samples(sdf_filename, self.subsample)

        if self.is_train:
            # reset seed for random sampling training data (see https://github.com/pytorch/pytorch/issues/5059)
            np.random.seed( int(time.time()) + idx)
            id = np.random.randint(0, self.num_views)
        else:
            np.random.seed(idx)
            id = np.random.randint(0, self.num_views)

        view_id = '{0:02d}'.format(id)

        image_folder = os.path.join(self.data_source, 'chairs', 'renders', mesh_name, 'naive_mad', 'base')
        image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
        image_files.sort()
        image_filename = image_files[id]
        RGBA = unpack_images(image_filename)

        sil_folder = os.path.join(self.data_source, 'chairs', 'renders', mesh_name, 'sil_mad', 'base')
        sil_files = glob.glob(os.path.join(sil_folder, '*.png'))
        sil_files.sort()
        sil_filename = sil_files[id]
        sil = unpack_images(sil_filename)
        sil = 1 - sil
        sil = sil[0]

        azi_list = [0, 135, 180, 225, 270, 315, 45, 90]
        azi = azi_list[id]

        # fetch cameras
        metadata_filename = os.path.join(self.img_source, mesh_name, 'camera', 'cameras.npz')
        intrinsic, extrinsic = get_camera_matrices(metadata_filename, id)

        return sdf_samples, RGBA, sil, intrinsic, azi, mesh_name


class RGBA2SDF_mask(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)
        # self.img_source = '/vol/research/ycau/shapeNet/03001627'

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        mesh_name = self.npyfiles[idx].split(".npz")[0].split('/')[-1]
        # fetch sdf samples
        sdf_filename = self.npyfiles[idx]
        sdf_samples = unpack_sdf_samples(sdf_filename, self.subsample)

        if self.is_train:
            # reset seed for random sampling training data (see https://github.com/pytorch/pytorch/issues/5059)
            np.random.seed( int(time.time()) + idx)
            id = np.random.randint(0, self.num_views)
        else:
            np.random.seed(idx)
            id = np.random.randint(0, self.num_views)

        view_id = '{0:02d}'.format(id)

        image_folder = os.path.join(self.data_source, 'chairs', 'renders', mesh_name, 'naive_mad', 'base')
        image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
        image_files.sort()
        image_filename = image_files[id]
        RGBA = unpack_images(image_filename)

        sil_folder = os.path.join(self.data_source, 'chairs', 'renders', mesh_name, 'sil_mad', 'base')
        sil_files = glob.glob(os.path.join(sil_folder, '*.png'))
        sil_files.sort()
        sil_filename = sil_files[id]
        sil = unpack_images(sil_filename)
        sil = 1 - sil
        sil = sil[0]

        azi_list = [0, 135, 180, 225, 270, 315, 45, 90]
        azi = azi_list[id]

        imask = torch.cat((RGBA, sil.unsqueeze(0)), dim=0)

        # # fetch cameras
        # metadata_filename = os.path.join(self.img_source, mesh_name, 'camera', 'cameras.npz')
        # intrinsic, extrinsic = get_camera_matrices(metadata_filename, id)

        return sdf_samples, imask, sil, azi, mesh_name




class RGBA2SDF_2SKETCH(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)
        self.img_source = '/vol/research/ycau/shapeNet/03001627'

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        mesh_name = self.npyfiles[idx].split(".npz")[0].split('/')[-1]
        # fetch sdf samples
        sdf_filename = self.npyfiles[idx]
        sdf_samples = unpack_sdf_samples(sdf_filename, self.subsample)

        if self.is_train:
            # reset seed for random sampling training data (see https://github.com/pytorch/pytorch/issues/5059)
            np.random.seed( int(time.time()) + idx)
            id1 = np.random.randint(0, self.num_views)
            id2 = np.random.randint(0, self.num_views)
            while id1==id2:
                id2 = np.random.randint(0, self.num_views)
        else:
            np.random.seed(idx)
            id1 = np.random.randint(0, self.num_views)
            id2 = np.random.randint(0, self.num_views)
            while id1==id2:
                id2 = np.random.randint(0, self.num_views)

        # view_id = '{0:02d}'.format(id)

        image_folder = os.path.join(self.data_source, 'chairs', 'renders', mesh_name, 'naive_mad', 'base')
        image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
        image_files.sort()
        image_filename1 = image_files[id1]
        RGBA1 = unpack_images(image_filename1)
        image_filename2 = image_files[id2]
        RGBA2 = unpack_images(image_filename2)
        # import pdb; pdb.set_trace()
        RGBA = torch.cat((RGBA1, RGBA2), dim=0)
        # RGBA = torch.stack((RGBA1, RGBA2), dim=0)

        sil_folder = os.path.join(self.data_source, 'chairs', 'renders', mesh_name, 'sil_mad', 'base')
        sil_files = glob.glob(os.path.join(sil_folder, '*.png'))
        sil_files.sort()
        sil_filename1 = sil_files[id1]
        sil1 = unpack_images(sil_filename1)
        sil1 = (1 - sil1)[0]
        sil_filename2 = sil_files[id2]
        sil2 = unpack_images(sil_filename2)
        sil2 = (1 - sil2)[0]
        sil = torch.cat((sil1.unsqueeze(0), sil2.unsqueeze(0)), dim=0)


        azi_list = [0, 135, 180, 225, 270, 315, 45, 90]
        azi1 = azi_list[id1]
        azi2 = azi_list[id2]

        # fetch cameras
        metadata_filename = os.path.join(self.img_source, mesh_name, 'camera', 'cameras.npz')
        # intrinsic, extrinsic = get_camera_matrices(metadata_filename, id)

        return sdf_samples, RGBA, sil, azi2, azi1, mesh_name
