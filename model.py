
import os, sys, inspect
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from collections import namedtuple


import util
from projection import Projection
from pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
import pytorch_utils as pt_utils


def get_model(num_classes, input_channels=0, use_xyz=True, bn=True):
    return Pointnet2MSG(
        num_classes=num_classes,
        input_channels=input_channels,
        use_xyz=use_xyz,
        bn=bn
    )

NPOINTS = [1024, 256, 64, 16]
RADIUS = [[0.05, 0.1], [0.1, 0.2], [0.2, 0.4], [0.4, 0.8]]
NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]
MLPS = [[[16, 16, 32], [32, 32, 64]], [[64, 64, 128], [64, 96, 128]],
        [[128, 196, 256], [128, 196, 256]], [[256, 256, 512], [256, 384, 512]]]
#FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]]
FP_MLPS = [[128, 128], [192, 192], [512, 512], [1024, 1024]]
CLS_FC = [128]
DP_RATIO = 0.5

def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "acc"])

    def model_fn(model, data, imageft, proj_ind_3d, proj_ind_2d, epoch=0, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)

            preds = model(inputs, imageft, proj_ind_3d, proj_ind_2d)
            loss = criterion(preds.view(labels.numel(), -1), labels.view(-1))

            _, classes = torch.max(preds, -1)

            acc = (classes[(classes*labels)>0] == labels[(classes*labels)>0]).float().sum() / labels[labels>0].numel()

            miou = list()
            if eval:
                for c in range(20):
                    mask_label = (labels == c+1)
                    mask_pred = (classes == c+1)

                    true_pos = ((mask_label * mask_pred).sum()).float()
                    false_pos = ((torch.ones(labels.shape).cuda() - mask_label.float()) * mask_pred.float()).sum()
                    false_neg = ((torch.ones(labels.shape).cuda() - mask_label.float()) * (torch.ones(labels.shape).cuda() - mask_pred.float())).sum()

                    miou.append((true_pos, true_pos + false_pos + false_neg))

                return ModelReturn(preds, loss, {"acc": acc.item(), "loss": loss.item(), "miou": miou})

            return ModelReturn(preds, loss, {"acc": acc.item(), "loss": loss.item()})

    return model_fn


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
        bn=True

        # reduce feature size to 32
        self.conv = nn.Conv2d(128, 32, 1)

        # pooling across num_images point clouds
        self.pooling = nn.MaxPool1d(kernel_size=self.num_images)

        # pointnet++
        # set abstraction (SA) layers
        self.SA_modules = nn.ModuleList()
        self.SA_modules_features = nn.ModuleList()
        self.SA_modules_geom = nn.ModuleList()
        channel_in = input_channels
        channel_in_geom = 0
        skip_channel_list = [input_channels]
        skip_channel_list_fused = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            # added for concatenation of geometry and feature point clouds
            mlps_geom = mlps.copy()
            channel_out_geom = channel_out
            for idx in range(mlps_geom.__len__()):
                mlps_geom[idx] = [channel_in_geom] + mlps_geom[idx]
                channel_out_geom += mlps_geom[idx][-1]
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            print(mlps)
            #self.SA_modules.append(
            #    PointnetSAModuleMSG(
            #        npoint=NPOINTS[k],
            #        radii=RADIUS[k],
            #        nsamples=NSAMPLE[k],
            #        mlps=mlps,
            #        use_xyz=use_xyz,
            #        bn=bn
            #    )
            #)

            self.SA_modules_geom.append(
                 PointnetSAModuleMSG(
                     npoint=NPOINTS[k],
                     radii=RADIUS[k],
                     nsamples=NSAMPLE[k],
                     mlps=mlps_geom,
                     use_xyz=use_xyz,
                     bn=bn
                 )
             )

            self.SA_modules_features.append(
                 PointnetSAModuleMSG(
                     npoint=NPOINTS[k],
                     radii=RADIUS[k],
                     nsamples=NSAMPLE[k],
                     mlps=mlps,
                     use_xyz=False,
                     bn=bn
                 )
             )
            skip_channel_list.append(channel_out)
            skip_channel_list_fused.append(channel_out + channel_out_geom)
            channel_in = channel_out
            channel_in_geom = channel_out_geom
            channel_in_fused = channel_out + channel_out_geom

        # feature propagation layer
        self.FP_modules = nn.ModuleList()
        self.FP_modules_fused = nn.ModuleList()

        #for k in range(FP_MLPS.__len__()):
        #    pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
        #    self.FP_modules.append(
        #        PointnetFPModule(
        #            mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k],
        #            bn=bn
        #        )
        #    )
#

        FP_MLPS_fused = [[256],[256],[512]]

        for k in range(FP_MLPS.__len__()):
             print(k)
             if(k+1 < len(FP_MLPS)):
                 print(FP_MLPS[k+1][-1], "<------------")
             pre_channel_fused = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out + channel_out_geom

             #if(k == 1):
             #    pre_channel_fused = 256-96
             #if(k == 2):
             #    pre_channel_fused = 256
             #if(pre_channel_fused == 2048):
             #    pre_channel_fused = 512

             print([pre_channel_fused + skip_channel_list_fused[k]] + FP_MLPS[k], "<-- FP")
             self.FP_modules_fused.append(
                 PointnetFPModule(
                     mlp=[pre_channel_fused + skip_channel_list_fused[k]] + FP_MLPS[k],
                     bn=bn
                 )
             )

        # classifier
        cls_layers = []
        pre_channel = FP_MLPS[0][-1]
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, CLS_FC[k], bn=bn))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, num_classes, activation=None, bn=bn))
        cls_layers.insert(1, nn.Dropout(0.5))
        self.cls_layer = nn.Sequential(*cls_layers)

        #self.fuse_layer_geom0 = nn.Linear(4096 * 6, 4096 * 3, bias=True)
        #self.fuse_layer_geom1 = nn.Linear(1024 * 6, 1024 * 3, bias=True)
        #self.fuse_layer_geom2 = nn.Linear(256 * 6, 256 * 3, bias=True)
        #self.fuse_layer_geom3 = nn.Linear(64 * 6, 64 * 3, bias=True)
        #self.fuse_layer_geom4 = nn.Linear(16 * 6, 16 * 3, bias=True)

        #self.fuse_layer_feat1 = nn.Linear(96 * 2048, 96 * 1024, bias=True)
        #self.fuse_layer_feat2 = nn.Linear(256 * 512, 256 * 256, bias=True)
        #self.fuse_layer_feat3 = nn.Linear(128 * 512, 512 * 64, bias=True)
        #self.fuse_layer_feat4 = nn.Linear(1024 * 32, 1024 * 16, bias=True)


    def _break_up_pc(self, pc):
        r"""
        Breaks point cloud up into coordinates (xyz) and features

        :param pc: Variable(torch.cuda.FloatTensor), (B, N, 3 + input_channels)
        :return: xyz (B, N, 3), features (B, N, input_channels)
        """
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, point_cloud, image_features, projection_indices_3d, projection_indices_2d):
        r"""
        forward pass of 3d model on fused input of features and geometry

        :param point_cloud: shape: (batch_size*num_images, num_input_channels, num_points_sample)
        :param image_features: shape: (batch_size*num_images, num_input_channels, proj_image_dims[0], proj_image_dims[1])
        :param projection_indices_3d: shape: (batch_size*num_images, num_points_sample)
        :param projection_indices_2d: shape: (batch_size*num_images, num_points_sample)
        :return: output of network after classifier
        """
        # projection_indices_3d (batch_size * num_images, num_points_sample + 1)
        assert len(point_cloud.shape) == 3 and len(image_features.shape) == 4
        batch_size = point_cloud.shape[0]
        num_points = point_cloud.shape[1] # number of points in sample. do we need number of points in whole scene?
        num_images = projection_indices_3d.shape[0] // batch_size

        # reduce number of feature channels

        image_features = self.conv(image_features)

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

        fuse_at_beginning = False

        if fuse_at_beginning:
            concatenated_cloud = torch.cat([point_cloud, image_features], 2)

            # split point cloud into coordinates and features
            xyz, features = self._break_up_pc(concatenated_cloud)
            l_xyz, l_features = [xyz], [features]
            # set abstraction
            for i in range(len(self.SA_modules)):
                li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
                l_xyz.append(li_xyz)
                l_features.append(li_features)

            # feature propagation
            for i in range(-1, -(len(self.FP_modules) + 1), -1):
                l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
                )

            # classifier
            pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()  # (B, N, num_classes)

        # fuse after set abstraction (before 'upsampling')
        else:
            xyz_geom, features_geom = self._break_up_pc(point_cloud)
            l_xyz_geom, l_features_geom = [xyz_geom], [features_geom]
            # set abstraction
            for i in range(len(self.SA_modules_geom)):
                li_xyz_geom, li_features_geom = self.SA_modules_geom[i](l_xyz_geom[i], l_features_geom[i])

                l_xyz_geom.append(li_xyz_geom)
                l_features_geom.append(li_features_geom)


            xyz, features = self._break_up_pc(image_features)
            l_xyz, l_features = [xyz], [features]

            l_xyz, l_features = [point_cloud.contiguous()], [image_features.transpose(1,2).contiguous()]
            # set abstraction
            for i in range(len(self.SA_modules_features)):
                li_xyz, li_features = self.SA_modules_features[i](l_xyz[i], l_features[i])
                l_xyz.append(li_xyz)
                l_features.append(li_features)

            #l_features.append(l_features_geom)
            #l_xyz.append(l_xyz_geom)
            l_xyz_fused = list()
            l_features_fused = list()

            for i in range(len(l_xyz)):
                l_xyz_fused.append(torch.cat((l_xyz_geom[i], l_xyz[i]), dim=2))
                if(i > 0):
                    l_features_fused.append(torch.cat((l_features_geom[i], l_features[i]),dim=1))
                else:
                    l_features_fused.append(l_features[i])





            ## Fusion of features and geometry
            #l_xyz_fused[0] = nn.ReLU(self.fuse_layer_geom0(l_xyz_fused[0].view(l_xyz_fused[0].shape, -1)))
            #l_xyz_fused[1] = nn.ReLU(self.fuse_layer_geom1(l_xyz_fused[1].view(l_xyz_fused[1].shape, -1)))
            #l_xyz_fused[2] = nn.ReLU(self.fuse_layer_geom2(l_xyz_fused[2].view(l_xyz_fused[2].shape, -1)))
            #l_xyz_fused[3] = nn.ReLU(self.fuse_layer_geom3(l_xyz_fused[3].view(l_xyz_fused[3].shape, -1)))
            #l_xyz_fused[4] = nn.ReLU(self.fuse_layer_geom4(l_xyz_fused[4].view(l_xyz_fused[4].shape, -1)))
            #l_features_fused[0] = l_features_fused[0]
            #l_features_fused[1] = nn.ReLU(self.fuse_layer_feat1(l_xyz_fused[1].view(l_xyz_fused[1].shape, -1)))
            #l_features_fused[2] = nn.ReLU(self.fuse_layer_feat2(l_xyz_fused[2].view(l_xyz_fused[2].shape, -1)))
            #l_features_fused[3] = nn.ReLU(self.fuse_layer_feat3(l_xyz_fused[3].view(l_xyz_fused[3].shape, -1)))
            #l_features_fused[4] = nn.ReLU(self.fuse_layer_feat4(l_xyz_fused[4].view(l_xyz_fused[4].shape, -1)))



            # feature propagation
            for i in range(-1, -(len(self.FP_modules_fused) + 1), -1):
                l_features[i - 1] = self.FP_modules_fused[i](
                    l_xyz_fused[i - 1], l_xyz_fused[i], l_features_fused[i - 1], l_features_fused[i]
                )

            # classifier
            pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()  # (B, N, num_classes)

        return pred_cls