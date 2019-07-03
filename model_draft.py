
import os, sys, inspect
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

import util
# import etw_pytorch_utils as pt_utils
from projection import Projection
# from pointnet2_modules import PointnetSAModule, PointnetFPModule
# from pointnet2.utils.pointnet2_modules import PointnetSAModule, PointnetFPModule

# z-y-x coordinates
class Model2d3d(nn.Module):

    def __init__(self, num_classes, num_images, input_channels, intrinsic, image_dims, depth_min, depth_max, accuracy):
        # added input_channels (should be 128 from 2d features)
        # deleted grid_dims
        super(Model2d3d, self).__init__()
        self.num_classes = num_classes
        self.num_images = num_images # for pooling
        self.intrinsic = intrinsic # for projection
        self.image_dims = image_dims # for projection
        self.depth_min = depth_min # for projection
        self.depth_max = depth_max # for projection
        self.accuracy = accuracy # originally: voxel size
        # added for pointnet++ (Whether or not to use the xyz position of a point as a feature)
        use_xyz = True

        # pooling across num_images point clouds
        self.pooling = nn.MaxPool1d(kernel_size=self.num_images)

    def _break_up_pc(self, pc):
        r"""
        Breaks point cloud up into coordinates (xyz) and features

        :param pc: Variable(torch.cuda.FloatTensor), (B, N, 3 + input_channels)
        :return: xyz (B, N, 3), features (B, N, input_channels)
        """
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, point_cloud, image_features, projection_indices_3d, projection_indices_2d):
        # projection_indices_3d (batch_size * num_images, num_points_sample + 1)
        assert len(point_cloud.shape) == 3 and len(image_features.shape) == 4
        batch_size = point_cloud.shape[0]
        num_points = point_cloud.shape[1] # number of points in sample. do we need number of points in whole scene?
        num_images = projection_indices_3d.shape[0] // batch_size

        # project 2d to 3d
        image_features = [Projection.apply(ft, ind3d, ind2d, num_points) for ft, ind3d, ind2d in zip(image_features, projection_indices_3d, projection_indices_2d)]
        image_features = torch.stack(image_features, dim=2) # (input_channels, num_points_sample, batch_size*num_images)

        # reshape to max pool over features
        sz = image_features.shape
        image_features = image_features.view(sz[0], -1, batch_size * num_images)
        # input size of max pooling: (batch, num_points, 3 + feature_channels, num_images)
        if num_images == self.num_images:
            image_features = self.pooling(image_features)
        else:
            image_features = nn.MaxPool1d(kernel_size=num_images)(image_features)
        image_features = image_features.view(sz[0], sz[1], batch_size)
        # image_features = image_features.permute(2, 0, 1) 3dmv format (batch_size, input_channels, num_points_sample)
        image_features = image_features.permute(2, 1, 0) # (batch_size, num_points_sample, input_channels)

        # pointnet++ on geometry and features,
        # TODO split pointnet++ and process geometry and features separately in the beginning
        concatenated_cloud = torch.cat([point_cloud, image_features], 2)

        # classifier
        return concatenated_cloud