
import torch
import numpy as np
import os
import argparse
from scipy import misc

import util
from data_util import resize_crop_image
from model import Model2d3d
from projection import ProjectionHelper

# initialize model and Projection Helper
proj_image_dims = [41, 32]
intrinsic = util.make_intrinsic(577.870605, 577.870605, 319.5, 239.5) # affine transformation from image plane to pixel coords
intrinsic = util.adjust_intrinsic(intrinsic, [640, 480], proj_image_dims)

projection = ProjectionHelper(intrinsic, 0.4, 4.0, proj_image_dims, 0.05)
model = Model2d3d(42, 3, 128, intrinsic, proj_image_dims, 0.4, 4.0, 0.05)
model = model.cuda()

# get point cloud
input = torch.Tensor(np.load('/media/lorenzlamm/My Book/pointnet2/scannet/preprocessing/scannet_scenes/scene0000_00.npy')).cuda()
point_cloud = input[:, :3]
num_points = point_cloud.shape[0]

batch_size = 2
num_images = 3
num_points_sample = 8192
point_batch = point_cloud.new(batch_size*num_images, num_points_sample, 3).cuda()
for i in range(batch_size*num_images):
  point_batch[i] = point_cloud[((i)*num_points_sample):((i+1)*num_points_sample),:]
point_batch[5] = point_batch[4]
imageft = torch.ones(batch_size*num_images, 128, proj_image_dims[0], proj_image_dims[1])


# get projection indices

# get depth_image
# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--data_path_2d',
        default='/media/lorenzlamm/My Book/Scannet/out_images',
        help='path to 2d train data')
opt = parser.parse_args()
print(opt)


depth_images = torch.cuda.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
# depth_images = torch.cuda.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])

# load_frames_multi
scan_name = 'scene0000_00'
frame_id = 0
depth_file = os.path.join(opt.data_path_2d, scan_name, 'depth', str(frame_id) + '.png')
depth_image_dims = [depth_images.shape[2], depth_images.shape[1]]

# load_depth_label_pose
depth_image = misc.imread(depth_file)
# preprocess
depth_image = resize_crop_image(depth_image, depth_image_dims) # resize to proj_iamge (features), i.e. 32x14
depth_image = depth_image.astype(np.float32) / 1000.0

depth_image = torch.from_numpy(depth_image)
depth_image_batch = depth_image.new(batch_size*num_images, proj_image_dims[1], proj_image_dims[0])
for i in range(batch_size*num_images):
    depth_image_batch[i] = depth_image

# get camera_pose
pose_file = os.path.join(opt.data_path_2d, scan_name, 'pose', str(frame_id) + '.txt')
lines = open(pose_file).read().splitlines()
assert len(lines) == 4
lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
camera_pose = torch.from_numpy(np.asarray(lines).astype(np.float32))
camera_pose_batch = camera_pose.new(batch_size*num_images, 4, 4).cuda()
for i in range(batch_size*num_images):
    camera_pose_batch[i] = camera_pose

# compute projection mapping
print(point_batch[4])
print(point_batch[4].shape)
print(point_batch[5].shape)

print(depth_image_batch[4])
print(camera_pose_batch[4])
proj_mapping_4 = projection.compute_projection(point_batch[4], depth_image_batch[4], camera_pose_batch[4], num_points_sample)
proj_mapping_5 = projection.compute_projection(point_batch[4], depth_image_batch[5], camera_pose_batch[5], num_points_sample)
proj_mapping = [projection.compute_projection(d, c, t, num_points_sample) for d, c, t in zip(point_batch, depth_image_batch, camera_pose_batch)]

proj_mapping = list(zip(*proj_mapping))
proj_ind_3d = torch.stack(proj_mapping[0]) # lin_indices_3d
proj_ind_2d = torch.stack(proj_mapping[1]) # lin_indices_2d


# forward pass
output = model(point_batch.cuda(), imageft.cuda(), torch.autograd.Variable(proj_ind_3d).cuda(), torch.autograd.Variable(proj_ind_2d).cuda())  # same inputs as forward fct