import pickle
import os
import sys
import numpy as np
import pc_util
import scene_util
import torch
import util
import argparse
from projection import ProjectionHelper
from scipy.spatial import ConvexHull
import h5py
import random

def pnt_in_pointcloud(points, new_pt):
    print points
    hull = ConvexHull(points)
    new_pts = points + new_pt
    print new_pts
    new_hull = ConvexHull(new_pts)
    print hull == new_hull
    print hull
    if hull == new_hull:
        return True
    else:
        return False




###############
# The following parameters were copied from 3DMV and are needed for the computation of camera intrinsics
# and projections and frustum bounds.

# params
parser = argparse.ArgumentParser()

# 2d/3d
parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size (in meters)')
parser.add_argument('--grid_dimX', type=int, default=31, help='3d grid dim x')
parser.add_argument('--grid_dimY', type=int, default=31, help='3d grid dim y')
parser.add_argument('--grid_dimZ', type=int, default=62, help='3d grid dim z')
parser.add_argument('--depth_min', type=float, default=0.4, help='min depth (in meters)')
parser.add_argument('--depth_max', type=float, default=4.0, help='max depth (in meters)')
# scannet intrinsic params
parser.add_argument('--intrinsic_image_width', type=int, default=640, help='2d image width')
parser.add_argument('--intrinsic_image_height', type=int, default=480, help='2d image height')
parser.add_argument('--fx', type=float, default=577.870605, help='intrinsics')
parser.add_argument('--fy', type=float, default=577.870605, help='intrinsics')
parser.add_argument('--mx', type=float, default=319.5, help='intrinsics')
parser.add_argument('--my', type=float, default=239.5, help='intrinsics')

parser.set_defaults(use_proxy_loss=False)
opt = parser.parse_args()

# create camera intrinsics
input_image_dims = [328, 256]
proj_image_dims = [41, 32]

intrinsic = util.make_intrinsic(opt.fx, opt.fy, opt.mx, opt.my)
intrinsic = util.adjust_intrinsic(intrinsic, [opt.intrinsic_image_width, opt.intrinsic_image_height], proj_image_dims)
intrinsic = intrinsic.cuda()
grid_dims = [opt.grid_dimX, opt.grid_dimY, opt.grid_dimZ]




def load_pose(filename):
    # from 3DMV: loads the pose matrix from the pose text file and returns it
    pose = torch.Tensor(4, 4)
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
    return torch.from_numpy(np.asarray(lines).astype(np.float32))


def findCorrespondingImages(chunksPath, posesPath, outPath, numImgs=3, npoints=8192):
    # finds the 3 (or more) images with the highest number of points in their frustum
    # and stores the result in a .hdf5 file
    # input:    chunksPath: Path to where the precomputed scene chunks are stored
    #           posesPath: Path to where the precomputed poses are stored
    #           numImgs: number of images to be used (default is 3)
    #           npoints: number of points in a point cloud
    # output:   Nothing; saves points, labels and poses into .hdf5 file

    print "Finding image correspondences"
    projection = ProjectionHelper(intrinsic, opt.depth_min, opt.depth_max, proj_image_dims, grid_dims, opt.voxel_size)

    fileList = list()
    for file in os.listdir(chunksPath):
        if file.endswith(".npy") and not file.startswith("006") and not file.startswith("007"):
            scene = file[:-4]
            fileList.append(scene)
    print fileList
    for scene in fileList:
        poseDict = {}
        print scene
        data = np.load(os.path.join(chunksPath,  scene + ".npy"))
        scene_points = data[:, :3]
        semantic_labels = data[:, 3]
        large_scene = scene[:7] # name of whole scene, e.g. 0000_01
        posesPathScene = os.path.join(posesPath, "scene"+large_scene, "pose")
        for poseFile in os.listdir(posesPathScene):
            pose = load_pose(os.path.join(posesPathScene,poseFile))
            corners = projection.compute_frustum_corners(pose)[:, :3, 0]
            normals = projection.compute_frustum_normals(corners)
            num_valid_points = projection.points_in_frustum(corners.double(), normals.double(), torch.DoubleTensor(scene_points))
            poseDict[poseFile[:-4]] = num_valid_points

        poseList = list()
        for i in range(numImgs):
            maximum = max(poseDict, key=poseDict.get)
            poseList.append(maximum)
            del poseDict[maximum]

        h5file = h5py.File(os.path.join(outPath, scene + ".hdf5"), "w")
        dset = h5file.create_dataset("points", (npoints, 3), data=scene_points)
        dset = h5file.create_dataset("labels", (npoints,), data=semantic_labels)
        dset = h5file.create_dataset("corresponding_images", (numImgs,), data=poseList)
        h5file.close()


def save_Npy_as_hdf5(chunksPath, out_path, npoints = 8192):
    # processing function for training only pointnet (without images)
    # converts numpy files into hdf5 files
    # these can then be converted to the right format with combine_to_final_data, where the flag  "with_images" should be set to false
    # train/val/test split : scenes 0-599 for training
    fileList = list()
    for file in os.listdir(chunksPath):
        if file.endswith(".npy") and not file.startswith("06") and not file.startswith("07"):
            scene = file[:-4]
            fileList.append(scene)
    for scene in fileList:
        print scene
        data = np.load(os.path.join(chunkPath, scene + ".npy"))
        scene_points = data[:, :3]
        semantic_labels = data[:, 3]
        h5file = h5py.File(os.path.join(out_path, scene + ".hdf5"), "w")
        dset = h5file.create_dataset("points", (npoints, 3), data=scene_points)
        dset = h5file.create_dataset("labels", (npoints,), data=semantic_labels)
        h5file.close()


#findCorrespondingImages("out_scenes", "2doutputs", "out_poses")

def findPointCloudsTraining(scenesPath, outPath, npoints = 8192):
    # Computes the scenes that will be used for training, i.e. chunks of the whole scenes that contain enough
    # (at least npoints) points.
    # input:    scenesPath: path where .npy pointclouds of all scenes are stored
    #           outPath: path where resulting scene chunks should be saved
    #           npoints: number of points in a pointcloud chunk (also miniumum number for a scene chunk to be kept)
    # output: None (scenes will be stored)


    fileList = list()
    for file in os.listdir(scenesPath):
        if file.endswith(".npy"):
            scene = file[-11:-4]
            fileList.append(scene)
    for scene in fileList:
        print scene
        data = np.load(os.path.join(scenesPath, "scene" + scene + ".npy"))
        whole_scene_points = data[:, :3]
        semantic_labels = data[:, 7]
        coordmax = np.max(whole_scene_points, axis=0)
        coordmin = np.min(whole_scene_points, axis=0)

        sliceCounter = 0
        for x in np.arange(coordmin[0] + 0.75, coordmax[0], 1.0):
            for y in np.arange(coordmin[1] - 0.75, coordmax[1], 1.0):
                if x > coordmax[0] - 0.75:
                    x = coordmax[0] - 0.75
                if y > coordmax[1] - 0.75:
                    y = coordmax[1] - 0.75

                curcenter = np.array([x, y, 1.5])
                curmin = curcenter - [0.75, 0.75, 1.5]
                curmax = curcenter + [0.75, 0.75, 1.5]
                curmin[2] = coordmin[2]
                curmax[2] = coordmax[2]
                curchoice = np.sum((whole_scene_points >= (curmin - 0.2)) * (whole_scene_points <= (curmax + 0.2)), axis=1) == 3
                if (np.sum(curchoice) <= npoints/2):
                    continue
                cur_point_set = whole_scene_points[curchoice, :]
                cur_semantic_labels = semantic_labels[curchoice]
                if len(cur_semantic_labels) == 0:
                    continue

                choice = np.random.choice(len(cur_semantic_labels), npoints, replace=True)
                scene_slice = cur_point_set[choice, :]
                scene_slice_sem = cur_semantic_labels[choice]
                whole_slice = np.concatenate((scene_slice, np.expand_dims(scene_slice_sem, axis = 1)), axis = 1)
                np.save(os.path.join(outPath, scene + "_" + str(sliceCounter) + ".npy"), whole_slice)
                sliceCounter += 1


#findPointCloudsTraining("pointclouds", "out_scenes")

def combine_to_final_data(in_path, out_path, num_scenes_per_file=1000, npoints=8192, with_images = True):
    fileList = list()
    for file in os.listdir(in_path):
        if file.endswith(".hdf5"):
            scene = file
            fileList.append(scene)
    save_number_of_scenes = open(os.path.join(out_path, "scene_count.txt"), "w")
    save_number_of_scenes.write(str(len(fileList)))
    print len(fileList)
    save_number_of_scenes.close()

    random.shuffle(fileList)
    file_count = 0
    scene_count = 0
    out_file = h5py.File(os.path.join(out_path, "scene_container_" + str(file_count)) + ".hdf5", 'w')
    in_data = np.array((npoints,3))
    in_labels = np.array(npoints)
    if(with_images):
        in_images = np.array(3)
    for i in range(len(fileList)):
        print(i)
        scene_count += 1
        in_file = h5py.File(os.path.join(in_path, fileList[i]))
        if(scene_count == 1):
            in_data = np.expand_dims(in_file['points'], 0)
            in_labels = np.expand_dims(in_file['labels'], 0)
            if(with_images):
                in_images = np.expand_dims(np.asarray(in_file['image_correspondences']), 0)
        else:
            in_data = np.append(in_data, np.expand_dims(in_file['points'], 0), axis=0)
            #in_data[scene_count] = in_file['points']
            in_labels = np.append(in_labels, np.expand_dims(in_file['labels'], 0), axis=0)
            #in_labels[scene_count] = in_file['labels']

            if (with_images):
                in_images = np.append(in_images, np.expand_dims(np.asarray(in_file['image_correspondences']), 0), axis=0)
                #in_images[scene_count] = np.asarray(in_file['image_correspondences'])
        if(scene_count == num_scenes_per_file):
            print(in_data.shape, in_labels.shape)
            dset = out_file.create_dataset("points", (num_scenes_per_file, npoints, 3), data=in_data)
            dset = out_file.create_dataset("labels", (num_scenes_per_file, npoints), data=in_labels)
            if(with_images):
                dset = out_file.create_dataset("corresponding_images", (num_scenes_per_file, 3), data=in_images)
            scene_count = 0
            file_count += 1
            out_file.close()
            out_file = h5py.File(os.path.join(out_path, "scene_container_" + str(file_count) + ".hdf5"), 'w')
    # save the remaining scenes
    dset = out_file.create_dataset("points", (scene_count, npoints, 3), data=in_data)
    dset = out_file.create_dataset("labels", (scene_count, npoints), data=in_labels)
    if (with_images):
        dset = out_file.create_dataset("corresponding_images", (scene_count, 3), data=in_images)

scenesPath = "/media/lorenzlamm/My Book/pointnet2/scannet/preprocessing/scannet_scenes"
chunkPath = "/media/lorenzlamm/Seagate Expansion Drive/ScanNet/sceneChunks2"
h5Path = "/media/lorenzlamm/Seagate Expansion Drive/ScanNet/h5files2"
finalPath = "/media/lorenzlamm/Seagate Expansion Drive/ScanNet/finalFiles3DMV_test"
imgPath = "/media/lorenzlamm/Seagate Expansion Drive/ScanNet/outimages"
outPath = "/media/lorenzlamm/Seagate Expansion Drive/ScanNet/correspondences"

#findCorrespondingImages(chunkPath, imgPath, outPath, numImgs=3, npoints=8192)
#findPointCloudsTraining(scenesPath, chunkPath, npoints=8192)
#save_Npy_as_hdf5(chunkPath, h5Path)
combine_to_final_data(outPath, finalPath, with_images=False)

for file in os.listdir(finalPath):
    print file

#test = h5py.File(os.path.join(finalPath, "scene_container_0"), 'r')
#print list(test.keys())
#print test['points'].shape
#print test['labels'].shape
