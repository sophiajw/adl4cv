import os
import argparse
#from scipy import misc
from scipy.ndimage import imread

import numpy as np
import torch
import h5py


from data.data_util import resize_crop_image
from projection_cpu import ProjectionHelper
from utils import util

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--data_path_2d',
        default='/Users/sophia/Documents/Studium/Mathematik/Master/AdvancedDeepLearning4ComputerVision/data/2doutput',
        help='path to 2d train data')
# scannet intrinsic params
parser.add_argument('--intrinsic_image_width', type=int, default=640, help='2d image width')
parser.add_argument('--intrinsic_image_height', type=int, default=480, help='2d image height')
parser.add_argument('--fx', type=float, default=577.870605, help='intrinsics')
parser.add_argument('--fy', type=float, default=577.870605, help='intrinsics')
parser.add_argument('--mx', type=float, default=319.5, help='intrinsics')
parser.add_argument('--my', type=float, default=239.5, help='intrinsics')
#2d/3d
parser.add_argument('--accuracy', type=float, default=0.1, help='voxel size (in meters)')
parser.add_argument('--depth_min', type=float, default=0.4, help='min depth (in meters)')
parser.add_argument('--depth_max', type=float, default=4.0, help='max depth (in meters)')

opt = parser.parse_args()
print(opt)

proj_image_dims = [240, 320]

def _load_data_file(name):
    f = h5py.File(name)
    data = f["points"][:]
    label = f["labels"][:]
    frames = f["corresponding_images"][:]
    return data, label, frames

# initialize projection class
# get intrinsic
intrinsic = util.make_intrinsic(opt.fx, opt.fy, opt.mx, opt.my)
intrinsic = util.adjust_intrinsic(intrinsic, [opt.intrinsic_image_width, opt.intrinsic_image_height], proj_image_dims)

projection = ProjectionHelper(intrinsic, opt.depth_min, opt.depth_max, proj_image_dims, opt.accuracy)

### compute_projection

points, label, frames = _load_data_file('/Users/sophia/Documents/Studium/Mathematik/Master/AdvancedDeepLearning4ComputerVision/data/train0600_00/train0600_00_15.hdf5')
points = torch.from_numpy(points)
points = points.type(torch.DoubleTensor)
label = torch.from_numpy(label).double()
## get depth_image
batch_size = 1
num_images = 1
depth_images = torch.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
# depth_images = torch.cuda.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
# load_frames_multi
scan_name = 'scene0600_00'
frame_id = 140
depth_file = os.path.join(opt.data_path_2d, scan_name, 'depth', str(frame_id) + '.png')
depth_image_dims = [depth_images.shape[2], depth_images.shape[1]]
# load_depth_label_pose
depth_image = imread(depth_file)
# preprocess
depth_image = resize_crop_image(depth_image, depth_image_dims) # resize to proj_iamge (features), i.e. 32x14
depth_image = depth_image.astype(np.float32) / 1000.0
depth_image = torch.from_numpy(depth_image)

# color image
color_images = torch.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
color_file = os.path.join(opt.data_path_2d, scan_name, 'color', str(frame_id) + '.jpg')
color_image = imread(color_file)
print(color_image.shape)
color_image = torch.from_numpy(color_image)
color_image = color_image.type(torch.DoubleTensor)



## get camera_pose
pose_file = os.path.join(opt.data_path_2d, scan_name, 'pose', str(frame_id) + '.txt')
lines = open(pose_file).read().splitlines()
assert len(lines) == 4
lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
camera_pose = torch.from_numpy(np.asarray(lines).astype(np.float32))

# load point cloud
np.savetxt('scenechunk_0600_00_15_labels.txt', torch.cat((points, label.unsqueeze(1)), dim=1), delimiter=',')
num_points = points.shape[0]
#
three_dim, two_dim = projection.compute_projection(points, depth_image, camera_pose, num_points)
#
print(points[three_dim[1:(three_dim[0]+1)]].type())
print(color_image.view((76800, 3)).type())
points_proj = torch.cat((points[three_dim[1:(three_dim[0]+1)]], color_image.view(76800, 3)[two_dim[1:(two_dim[0]+1)]]), dim=1)
print(points_proj.shape)
np.savetxt('test_projection_chunk_0600_00_15.txt', points_proj, delimiter=',')
#
#corner_coords = projection.compute_frustum_corners(camera_pose)
#np.savetxt('corner_coords_840.txt', corner_coords, delimiter=',')
# normals = projection.compute_frustum_normals(corner_coords)
# new_pt = projection.point_in_frustum(corner_coords, normals, corner_coords[4][:3].view(-1))


