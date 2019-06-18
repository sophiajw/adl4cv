
import numpy as np
import torch
from torch.autograd import Function

class ProjectionHelper():
    def __init__(self, intrinsic, depth_min, depth_max, image_dims, volume_dims, accuracy):
        self.intrinsic = intrinsic
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.image_dims = image_dims
        self.volume_dims = volume_dims
        self.accuracy = accuracy


    def depth_to_skeleton(self, ux, uy, depth):
        # 2D to 3D coordinates with depth (used in compute_frustum_bounds)
        x = (ux - self.intrinsic[0][2]) / self.intrinsic[0][0]
        y = (uy - self.intrinsic[1][2]) / self.intrinsic[1][1]
        return torch.Tensor([depth*x, depth*y, depth])


    def skeleton_to_depth(self, p):
        x = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        y = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        return torch.Tensor([x, y, p[2]])


    def compute_frustum_bounds(self, camera_to_world):
        # input: camera pose (torch.Size([4, 4]))
        # output: two points (x, y, z) (torch.Size([3])) that define the viewing frustum of the camera

        corner_points = camera_to_world.new(8, 4, 1).fill_(1)

        # pixel to camera
        # depth min
        corner_points[0][:3] = self.depth_to_skeleton(0, 0, self.depth_min).unsqueeze(1)
        corner_points[1][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_min).unsqueeze(1)
        corner_points[2][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_min).unsqueeze(1)
        corner_points[3][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_min).unsqueeze(1)
        # depth max
        corner_points[4][:3] = self.depth_to_skeleton(0, 0, self.depth_max).unsqueeze(1)
        corner_points[5][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_max).unsqueeze(1)
        corner_points[6][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_max).unsqueeze(1)
        corner_points[7][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_max).unsqueeze(1)
        print(corner_points)

        # camera to world
        corner_coords = torch.bmm(camera_to_world.repeat(8, 1, 1), corner_points)

        # choose the smallest (largest) values of x, y and z across the 8 points for bbox_min (bbox_max)
        bbox_min, _ = torch.min(corner_coords[:, :3, 0], 0)
        bbox_max, _ = torch.max(corner_coords[:, :3, 0], 0)
        return bbox_min, bbox_max


    def compute_projection(self, points, depth, camera_to_world, num_points):
        # input: depth map (size: proj_image), camera pose (4x4), number of points in our point cloud
        # maybe: initialize ProjectionHelper with num_points
        # output: correspondence of points to pixels

        # compute viewing frustum bounds
        world_to_camera = torch.inverse(camera_to_world)
        point_bound_min, point_bound_max = self.compute_frustum_bounds(camera_to_world)
        # TODO check if bounds are valid? positive and in correct range?
        # .cuda()

        # create list with all points and their coordinates
        # should just be the list imported from h5 file
        # ind_points = torch.arange(0, num_points, out = torch.LongTensor()) # .cuda()
        # coords = camera_to_world.new(4, ind_points.size(0))

        # dummy-list
        ind_points = torch.arange(0, num_points, out=torch.LongTensor())
        coords = camera_to_world.new(4, num_points)
        coords[:3, :] = torch.t(points)
        coords[3, :].fill_(1)

        # consider only points that lie in frustum bound
        mask_frustum_bounds = torch.ge(coords[0], point_bound_min[0]) * torch.ge(coords[1], point_bound_min[1]) * torch.ge(coords[2], point_bound_min[2])
        mask_frustum_bounds = mask_frustum_bounds * torch.lt(coords[0], point_bound_max[0]) * torch.lt(coords[1], point_bound_max[1]) * torch.lt(coords[2], point_bound_max[2])
        if not mask_frustum_bounds.any():
            return None
        ind_points = ind_points[mask_frustum_bounds]
        coords = coords[:, ind_points]

        # project world (coords) to camera
        camera = torch.mm(world_to_camera, coords)

        # project camera to image
        camera[0] = (camera[0] * self.intrinsic[0][0]) / camera[2] + self.intrinsic[0][2]
        camera[1] = (camera[1] * self.intrinsic[1][1]) / camera[2] + self.intrinsic[1][2]
        image = torch.round(camera).long()

        # keep points that are projected onto the image into the correct pixel range
        valid_ind_mask = torch.ge(image[0], 0) * torch.ge(image[1], 0) * torch.lt(image[0], self.image_dims[0]) * torch.lt(image[1], self.image_dims[1])
        if not valid_ind_mask.any():
            return None
        valid_image_ind_x = image[0][valid_ind_mask]
        valid_image_ind_y = image[1][valid_ind_mask]
        valid_image_ind = valid_image_ind_y * self.image_dims[0] + valid_image_ind_x

        # keep only points that are in the correct depth ranges (self.depth_min - self.depth_max)
        depth_vals = torch.index_select(depth.view(-1), 0, valid_image_ind)
        depth_mask = depth_vals.ge(self.depth_min) * depth_vals.le(self.depth_max) * torch.abs(depth_vals - camera[2][valid_ind_mask]).le(self.accuracy)
        # TODO torch.abs(...) necessary?
        if not depth_mask.any():
            return None

        # create two vectors for all considered points that establish 3d to 2d correspondence
        ind_update = ind_points[valid_ind_mask]
        ind_update = ind_update[depth_mask]
        indices_3d = ind_update.new(num_points + 1) # needs to be same size for all in batch... (first element has size)
        indices_2d = ind_update.new(num_points + 1) # needs to be same size for all in batch... (first element has size)
        indices_3d[0] = ind_update.shape[0]  # first entry: number of relevant entries (of points)
        indices_2d[0] = ind_update.shape[0]
        indices_3d[1:1 + indices_3d[0]] = ind_update  # indices of points
        indices_2d[1:1 + indices_2d[0]] = torch.index_select(valid_image_ind, 0, torch.nonzero(depth_mask)[:, 0])  # indices of corresponding pixels

        return indices_3d, indices_2d


# Inherit from Function
class Projection(Function):

    @staticmethod
    def forward(ctx, label, lin_indices_3d, lin_indices_2d, volume_dims):
        ctx.save_for_backward(lin_indices_3d, lin_indices_2d)
        num_label_ft = 1 if len(label.shape) == 2 else label.shape[0]
        output = label.new(num_label_ft, volume_dims[2], volume_dims[1], volume_dims[0]).fill_(0)
        num_ind = lin_indices_3d[0]
        if num_ind > 0:
            vals = torch.index_select(label.view(num_label_ft, -1), 1, lin_indices_2d[1:1+num_ind])
            output.view(num_label_ft, -1)[:, lin_indices_3d[1:1+num_ind]] = vals
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_label = grad_output.clone()
        num_ft = grad_output.shape[0]
        grad_label.data.resize_(num_ft, 32, 41)
        lin_indices_3d, lin_indices_2d = ctx.saved_variables
        num_ind = lin_indices_3d.data[0]
        vals = torch.index_select(grad_output.data.contiguous().view(num_ft, -1), 1, lin_indices_3d.data[1:1+num_ind])
        grad_label.data.view(num_ft, -1)[:, lin_indices_2d.data[1:1+num_ind]] = vals
        return grad_label, None, None, None

