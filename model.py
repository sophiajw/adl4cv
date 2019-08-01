
import torch
import torch.nn as nn

from projection import Projection
from pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
import pytorch_utils as pt_utils

# parameters needed for initialization of PointNet++ layers
NPOINTS = [1024, 256, 64, 16]
RADIUS = [[0.05, 0.1], [0.1, 0.2], [0.2, 0.4], [0.4, 0.8]]
NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]
MLPS = [[[16, 16, 32], [32, 32, 64]], [[64, 64, 128], [64, 96, 128]],
        [[128, 196, 256], [128, 196, 256]], [[256, 256, 512], [256, 384, 512]]]
FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]]
CLS_FC = [128]
DP_RATIO = 0.5


class Model2d3d(nn.Module):

    def __init__(self, num_classes, num_images, input_channels, intrinsic, image_dims, depth_min, depth_max, accuracy,
                 fusion = True, fuseAtPosition=2, fuse_no_ft_pn = False, pointnet_pointnet = False):
        """
        Initialization of our model with different fusing methods for feature and geometry point clouds in PointNet++.
        Default settings initialize our best model, i.e. fusion after two set abstraction layers.

        :param num_classes: (int) number of classes to predict. = 21 for Scannet
        :param num_images: (int) number of images considered per sample, default = 3
        :param input_channels: (int) number of feature channels used for PointNet++ input
        :param intrinsic: camera intrinsics of Scannet cameras
        :param image_dims: [int, int] 2D feature map dimensions
        :param depth_min: (float) min depth [m] of camera
        :param depth_max: (float) max depth [m] of camera
        :param accuracy: (float) accuracy for projection layer
        :param fusion: (boolean) Fuse in set abstraction layers of PointNet++?
        :param fuseAtPosition: (1, 2 or 4) Fuse after fuseAtPosition set abstraction layers
        :param fuse_no_ft_pn: (boolean) Process only geomtry point cloud with PointNet++?
        :param pointnet_pointnet: (boolean) Apply PointNet++ in all steps
        """
        super(Model2d3d, self).__init__()
        self.pointnet_pointnet = pointnet_pointnet
        self.fusion = fusion
        self.fuse_at_position = fuseAtPosition
        self.fuse_no_ft_pn = fuse_no_ft_pn
        if(self.fuse_no_ft_pn):
            self.fusion = False
        self.num_classes = num_classes
        self.num_images = num_images # for pooling
        self.intrinsic = intrinsic # for projection
        self.image_dims = image_dims # for projection
        self.depth_min = depth_min # for projection
        self.depth_max = depth_max # for projection
        self.accuracy = accuracy
        use_xyz = True # added for pointnet++ (Whether or not to use the xyz position of a point as a feature)
        bn=True

        # pooling across num_images point clouds
        self.pooling = nn.MaxPool1d(kernel_size=self.num_images)

        # pointnet++
        # set abstraction (SA) layers
        self.SA_modules = nn.ModuleList()
        self.SA_modules_features = nn.ModuleList()
        self.SA_modules_geom = nn.ModuleList()
        if self.pointnet_pointnet:
            self.SA_modules_concat = nn.ModuleList()
        if self.fuse_no_ft_pn:
            self.fuseConv = nn.Conv1d(256, 128, kernel_size=1)
        channel_in = input_channels
        if self.fusion or self.pointnet_pointnet:
            channel_in = 0
        channel_in_feat = input_channels
        skip_channel_list = [channel_in]
        skip_channel_list_feat = [channel_in_feat]
        skip_channel_list_fused = [input_channels]
        if self.pointnet_pointnet:
            channel_in_concat = 256
            skip_channel_list_concat = [channel_in_concat]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            if self.fusion or self.pointnet_pointnet:
                mlps_feat = MLPS[k].copy()
            if self.pointnet_pointnet:
                mlps_concat = MLPS[k].copy()
            channel_out = 0
            # added for concatenation of geometry and feature point clouds
            channel_out_geom = channel_out
            channel_out_feat = 0
            if self.pointnet_pointnet:
                channel_out_concat = 0
            if self.fusion:
                if k == fuseAtPosition:
                    channel_in += channel_in_feat

            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            if self.fusion or self.pointnet_pointnet:
                for idx in range(mlps.__len__()):
                    mlps_feat[idx] = [channel_in_feat] + mlps_feat[idx]
                    channel_out_feat += mlps_feat[idx][-1]
            if self.pointnet_pointnet:
                for idx in range(mlps.__len__()):
                    mlps_concat[idx] = [channel_in_concat] + mlps_concat[idx]
                    channel_out_concat += mlps_concat[idx][-1]
            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=NPOINTS[k],
                    radii=RADIUS[k],
                    nsamples=NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=use_xyz,
                    bn=bn
                )
            )

            if self.fusion or self.pointnet_pointnet:
                self.SA_modules_features.append(
                    PointnetSAModuleMSG(
                        npoint=NPOINTS[k],
                        radii=RADIUS[k],
                        nsamples=NSAMPLE[k],
                        mlps=mlps_feat,
                        use_xyz=False,
                        bn=bn
                    )
                )
            if self.pointnet_pointnet:
                self.SA_modules_concat.append(
                    PointnetSAModuleMSG(
                        npoint=NPOINTS[k],
                        radii=RADIUS[k],
                        nsamples=NSAMPLE[k],
                        mlps=mlps_concat,
                        use_xyz=False,
                        bn=bn
                    )
                )

            if self.pointnet_pointnet:
                skip_channel_list_concat.append(channel_out_concat)
            skip_channel_list.append(channel_out)
            skip_channel_list_feat.append(channel_out_feat)
            skip_channel_list_fused.append(channel_out + channel_out_geom)
            channel_in = channel_out
            if self.fusion or self.pointnet_pointnet:
                channel_in_feat = channel_out_feat
            if self.pointnet_pointnet:
                channel_in_concat = channel_out_concat

        # feature propagation layers
        self.FP_modules = nn.ModuleList()
        self.FP_modules_feat = nn.ModuleList()
        self.FP_modules_fused = nn.ModuleList()
        if self.pointnet_pointnet:
            self.FP_modules_concat = nn.ModuleList()

        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            if(self.fusion):
                if k == self.fuse_at_position:
                    self.FP_modules.append(
                        PointnetFPModule(
                            mlp=[pre_channel + skip_channel_list[k]*2] + FP_MLPS[k],
                            bn=bn
                        )
                    )
                elif self.fuse_at_position == 4 and k == FP_MLPS.__len__()-1:
                    self.FP_modules.append(
                        PointnetFPModule(
                            mlp=[pre_channel*2 + skip_channel_list[k]] + FP_MLPS[k],
                            bn=bn
                        )
                    )
                else:
                    self.FP_modules.append(
                        PointnetFPModule(
                            mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k],
                            bn=bn
                        )
                    )
            elif self.pointnet_pointnet:
                pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out_feat
                self.FP_modules_feat.append(
                    PointnetFPModule(
                        mlp=[pre_channel + skip_channel_list_feat[k]] + FP_MLPS[k],
                        bn=bn
                    )
                )
                pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out_concat
                self.FP_modules_concat.append(
                    PointnetFPModule(
                        mlp=[pre_channel + skip_channel_list_concat[k]] + FP_MLPS[k],
                        bn=bn
                    )
                )
                pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
                self.FP_modules.append(
                    PointnetFPModule(
                        mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k],
                        bn=bn
                    )
                )
            else:
                self.FP_modules.append(
                    PointnetFPModule(
                        mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k],
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

    def _break_up_pc(self, pc):
        """
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
        """
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
        image_features = image_features.permute(2, 1, 0) # shape: (batch_size, num_points_sample, input_channels)

        # pointnet++ on geometry and features,

        if self.fusion:
            # add coordinates to feature point cloud
            concatenated_cloud = torch.cat([point_cloud, image_features], 2)

            # feature stream
            # split point cloud into coordinates and features
            xyz, features = self._break_up_pc(concatenated_cloud)
            l_xyz_feat, l_features_feat = [xyz], [features]

            # set abstraction layers
            for i in range(self.fuse_at_position):
                li_xyz_feat, li_features_feat = self.SA_modules_features[i](l_xyz_feat[i], l_features_feat[i])
                l_xyz_feat.append(li_xyz_feat)
                l_features_feat.append(li_features_feat)

            # geometry stream
            # split point cloud into coordinates and features
            xyz, features = self._break_up_pc(point_cloud)
            l_xyz, l_features = [xyz], [features]

            # set abstraction
            for i in range(len(self.SA_modules)):
                li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
                l_xyz.append(li_xyz)
                l_features.append(li_features)

                # fuse feature and geometry streams
                if i == self.fuse_at_position-1:
                    l_features[-1] = torch.cat((l_features[-1], l_features_feat[-1]), 1)

            # feature propagation on fused streams
            for i in range(-1, -(len(self.FP_modules) + 1), -1):
                l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
                )

            # classifier
            pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()  # (B, N, num_classes)

        elif self.pointnet_pointnet:
            # add coordinates to feature point cloud
            concatenated_cloud = torch.cat([point_cloud, image_features], 2)

            # feature stream

            # split point cloud into coordinates and features
            xyz_feat, features_feat = self._break_up_pc(concatenated_cloud)
            l_xyz_feat, l_features_feat = [xyz_feat], [features_feat]

            # set abstraction layers
            for i in range(len(self.SA_modules_features)):
                li_xyz_feat, li_features_feat = self.SA_modules_features[i](l_xyz_feat[i], l_features_feat[i])
                l_xyz_feat.append(li_xyz_feat)
                l_features_feat.append(li_features_feat)

            # feature propagation
            for i in range(-1, -(len(self.FP_modules_feat) + 1), -1):
                l_features_feat[i - 1] = self.FP_modules_feat[i](
                    l_xyz_feat[i - 1], l_xyz_feat[i], l_features_feat[i - 1], l_features_feat[i]
                )

            # geometry stream

            # split point cloud into coordinates and features
            xyz, features = self._break_up_pc(point_cloud)
            l_xyz, l_features = [xyz], [features]

            # set abstraction layers
            for i in range(len(self.SA_modules)):
                li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
                l_xyz.append(li_xyz)
                l_features.append(li_features)

            # feature propagation
            for i in range(-1, -(len(self.FP_modules) + 1), -1):
                l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
                )

            # fuse feature and geomtetry stream and apply PointNet++
            concat_features = torch.cat((l_features[0],l_features_feat[0]), dim=1)
            l_xyz_concat, l_features_concat = [l_xyz_feat[0]], [concat_features]

            # set abstraction layers
            for i in range(len(self.SA_modules_concat)):
                li_xyz_concat, li_features_concat = self.SA_modules_concat[i](l_xyz_concat[i], l_features_concat[i])
                l_xyz_concat.append(li_xyz_concat)
                l_features_concat.append(li_features_concat)

            # feature propagation
            for i in range(-1, -(len(self.FP_modules_concat) + 1), -1):
                l_features_concat[i - 1] = self.FP_modules_concat[i](
                    l_xyz_concat[i - 1], l_xyz_concat[i], l_features_concat[i - 1], l_features_concat[i]
                )

            # classifier
            pred_cls = self.cls_layer(l_features_concat[0]).transpose(1, 2).contiguous()  # (B, N, num_classes)

        elif self.fuse_no_ft_pn:
            # Fusion of features together with features from Pointnet++ (extracted only from geometry)

            xyz, features = self._break_up_pc(point_cloud)
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

            # fuse feature and geometry
            l_features[0] = torch.cat((l_features[0], image_features.transpose(1,2)), dim=1)
            l_features[0] = nn.functional.relu((self.fuseConv(l_features[0])))

            # classifier
            pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()  # (B, N, num_classes)

        else:
            # concatenate feature and geometry point cloud directly and apply PointNet++
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

        return pred_cls
