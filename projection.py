
import numpy as np
import torch
from torch.autograd import Function


class ProjectionHelper():
    def __init__(self, intrinsic, depth_min, depth_max, image_dims, accuracy):
        self.intrinsic = intrinsic
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.image_dims = image_dims
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


    def compute_frustum_corners(self, camera_to_world):
        # input: camera pose (torch.Size([4, 4]))
        # output: coordinates of the corner points of the viewing frustum of the camera

        corner_points = camera_to_world.new(8, 4, 1).fill_(1)

        # image to camera
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

        # camera to world
        corner_coords = torch.bmm(camera_to_world.repeat(8, 1, 1), corner_points)

        return corner_coords

    def compute_frustum_normals(self, corner_coords):
        # input: coordinates of the corner points of the viewing frustum
        # output: normals of the 6 planes defining the viewing frustum
        normals = corner_coords.new(6, 3)

        # compute plane normals
        # front plane
        plane_vec1 = corner_coords[3][:3] - corner_coords[0][:3]
        plane_vec2 = corner_coords[1][:3] - corner_coords[0][:3]
        normals[0] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        # right side plane
        plane_vec1 = corner_coords[2][:3] - corner_coords[1][:3]
        plane_vec2 = corner_coords[5][:3] - corner_coords[1][:3]
        normals[1] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        # roof plane
        plane_vec1 = corner_coords[3][:3] - corner_coords[2][:3]
        plane_vec2 = corner_coords[6][:3] - corner_coords[2][:3]
        normals[2] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        # left side plane
        plane_vec1 = corner_coords[0][:3] - corner_coords[3][:3]
        plane_vec2 = corner_coords[7][:3] - corner_coords[3][:3]
        normals[3] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        # bottom plane
        plane_vec1 = corner_coords[1][:3] - corner_coords[0][:3]
        plane_vec2 = corner_coords[4][:3] - corner_coords[0][:3]
        normals[4] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        # back plane
        plane_vec1 = corner_coords[6][:3] - corner_coords[5][:3]
        plane_vec2 = corner_coords[4][:3] - corner_coords[5][:3]
        normals[5] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        return normals

    def point_in_frustum(self, corner_coords, normals, new_pt):
        # input: coordinates of points defining the frustum, normals defining the planes of the frustum
        #       (pointing inwards), new point (must be torch.Size([3]))
        # output: 1 or 0 whether new point is in viewing frustum or not

        # create vector from new_pt to the planes
        point_to_plane = corner_coords.new(6, 3)
        point_to_plane[0:3] = (new_pt - corner_coords[2][:3].view(-1)).repeat(3, 1)
        point_to_plane[3:6] = (new_pt - corner_coords[4][:3].view(-1)).repeat(3, 1)

        # check if the scalar product with the normals is positive
        for k, normal in enumerate(normals):
            if torch.round(torch.dot(normal, point_to_plane[k]) * 100) / (100) > 0:
                return 0

        return 1


    def compute_projection(self, points, depth, camera_to_world, num_points):
        # input: tensor containing all points of the point cloud, depth map (size: proj_image), camera pose (4x4),
        #          number of points in our point cloud
        # maybe: initialize ProjectionHelper with num_points
        # output: correspondence of points to pixels

        world_to_camera = torch.inverse(camera_to_world)

        # create 1-dim array with all indices and array with 4-dim coordinates x, y, z, 1 of points
        ind_points = torch.arange(0, num_points, out=torch.LongTensor())
        coords = camera_to_world.new(4, num_points)
        coords[:3, :] = torch.t(points)
        coords[3, :].fill_(1)

        # compute viewing frustum
        corner_coords = self.compute_frustum_corners(camera_to_world)
        normals = self.compute_frustum_normals(corner_coords)
        # .cuda()

        # check if points are in viewing frustum and only keep according indices
        mask_frustum_bounds = torch.ByteTensor(num_points)
        for k, point in enumerate(points):
            mask_frustum_bounds[k] = self.point_in_frustum(corner_coords, normals, point)

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
    def forward(ctx, label, lin_indices_3d, lin_indices_2d, num_points):
        r"""
        forward pass of backprojection for 2d features onto 3d points

        :param label: image features (shape: (num_input_channels, proj_image_dims[0], proj_image_dims[1]))
        :param lin_indices_3d: point indices from projection (shape: (num_input_channels, num_points_sample))
        :param lin_indices_2d: pixel indices from projection (shape: (num_input_channels, num_points_sample))
        :param num_points: number of points in one sample
        :return: array of points in sample with projected features (shape: (num_input_channels, num_points))
        """
        # label  = image_features (128, 41, 32)
        ctx.save_for_backward(lin_indices_3d, lin_indices_2d)
        num_label_ft = 1 if len(label.shape) == 2 else label.shape[0] # = num_input_channels

        output = label.new(num_label_ft, num_points).fill_(0)
        num_ind = lin_indices_3d[0]
        if num_ind > 0:
            # selects values from image_features at indices given by lin_indices_2d
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

