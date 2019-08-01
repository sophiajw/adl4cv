import os
import numpy as np
import torch
import util
import argparse
from projection import ProjectionHelper
import h5py
import time


"""
The following parameters were adapted from 3DMV and are needed for the computation of camera intrinsics
and projections and frustum bounds.
"""


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
intrinsic = intrinsic
grid_dims = [opt.grid_dimX, opt.grid_dimY, opt.grid_dimZ]




def load_pose(filename):
    """
    from 3DMV: loads the pose matrix from the pose text file and returns it
    :param filename: name of text file
    """
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
    return torch.from_numpy(np.asarray(lines).astype(np.float32))


def findPointCloudsSplit(split_path, scenesPath, outPath):
    """
    Computes the scenes that will be used for training, i.e. chunks of the whole scenes that contain enough
    (at least npoints) points.
    :param split_path: path to where .txt files with benchmark split are stored
    :param scenesPath: path where .npy pointclouds of all scenes are stored
    :param outPath: path where resulting scene chunks should be saved
    """

    for split in ["train", "test", "val"]:
        fileList = list() # List containing the names of numpy scene files

        # Splits are loaded from ScanNet v1 benchmark (since for v2 test data, we have no image data)
        with open(os.path.join(split_path, 'scannetv1_' + split + '.txt')) as f:
            for line in f:
                scene = line[:-1]
                fileList.append(scene)

        # Iterate through all scenes and compute scene chunks
        for scene in fileList:
            if(not os.path.isfile(os.path.join(scenesPath, scene + '.npy'))):
                print(scene, "was not processed correctly, skipping this one.")
                continue
            data = np.load(os.path.join(scenesPath, scene + ".npy"))
            whole_scene_points = data[:, :3]
            semantic_labels = data[:, 7]
            coordmax = np.max(whole_scene_points, axis=0)
            coordmin = np.min(whole_scene_points, axis=0)

            sliceCounter = 0

            # Slide through the scene with chunks of 1.5m x 1.5m x 3m
            # If the chunk contains at least 1024 points, we keep it.
            # We store all points and later, our data loader will sample from the scene chunks.
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
                    if (np.sum(curchoice) <= 1024):
                        continue
                    cur_point_set = whole_scene_points[curchoice, :]
                    cur_semantic_labels = semantic_labels[curchoice]
                    if len(cur_semantic_labels) == 0: #if we don't have any labeled points in frustum, continue
                        continue
                    scene_slice = cur_point_set
                    scene_slice_sem = cur_semantic_labels

                    # Store resulting scene chunks
                    whole_slice = np.concatenate((scene_slice, np.expand_dims(scene_slice_sem, axis = 1)), axis = 1)
                    np.save(os.path.join(outPath, split + scene + "_" + str(sliceCounter) + ".npy"), whole_slice)
                    sliceCounter += 1



def findPointCloudsTestWholeScene(scenesPath, outPath, scenenr = '0000_00'):
    """
    For visualization of test scenes, we slide through the whole scenes and compute the scene chunks with only little overlap
    Rest is similar to findPointCloudsSplit
    Difference: We keep the scene chunks if they contain at least one point
    :param scenesPath: path where .npy pointclouds of all scenes are stored
    :param outPath: path where resulting scene chunks should be saved
    :param npoints: number of points in a pointcloud chunk (also miniumum number for a scene chunk to be kept)
    :param scenenr: Number of scene which should be processed
    """
    fileList = list()
    for file in os.listdir(scenesPath):
        if file.endswith(".npy") and file.startswith('scene'+scenenr):
            scene = file[-11:-4]
            fileList.append(scene)
    for scene in fileList:
        print(scene, "<---")
        data = np.load(os.path.join(scenesPath, "scene" + scene + ".npy"))
        whole_scene_points = data[:, :3]
        semantic_labels = data[:, 7]
        coordmax = np.max(whole_scene_points, axis=0)
        coordmin = np.min(whole_scene_points, axis=0)
        sliceCounter = 0
        for x in np.arange(coordmin[0] + 0.75, coordmax[0], 1.35):
            for y in np.arange(coordmin[1] - 0.75, coordmax[1], 1.35):
                print(x,y)
                if x > coordmax[0] - 0.75:
                    x = coordmax[0] - 0.75
                if y > coordmax[1] - 0.75:
                    y = coordmax[1] - 0.75

                curcenter = np.array([x, y, 1.5])
                curmin = curcenter - [0.75, 0.75, 1.5]
                curmax = curcenter + [0.75, 0.75, 1.5]
                curmin[2] = coordmin[2]
                curmax[2] = coordmax[2]
                curchoice = np.sum((whole_scene_points >= (curmin - 0.2)) * (whole_scene_points <= (curmax + 0.2)),
                                   axis=1) == 3
                cur_point_set = whole_scene_points[curchoice, :]
                cur_semantic_labels = semantic_labels[curchoice]
                if len(cur_semantic_labels) == 0: #if we don't have any labeled points in frustum, continue
                    continue
                scene_slice = cur_point_set
                scene_slice_sem = cur_semantic_labels

                # Store resulting scene chunks, as well as according GT scene chunks
                whole_slice = np.concatenate((scene_slice, np.expand_dims(scene_slice_sem, axis=1)), axis=1)
                whole_slice_compare = np.concatenate((scene_slice, np.expand_dims(sliceCounter * np.ones(scene_slice.shape[0]), axis=1)), axis = 1)
                np.save(os.path.join(outPath, scene + "_" + str(sliceCounter) + ".npy"), whole_slice)
                np.save(os.path.join(outPath, "compare", scene + "_" + str(sliceCounter) + "_compare.npy"), whole_slice_compare)
                sliceCounter += 1


def findCorrespondingImages(chunksPath, posesPath, outPath, numImgs=3):
    """
    for each scene chunk, finds the 3 (or more) images with the highest number of points in their frustum
    and stores the result in a .hdf5 file
    :param chunksPath: Path to where the precomputed scene chunks are stored
    :param posesPath: Path to where the precomputed poses are stored
    :param outPath: Path to where scene chunks with corresponding images should be stored
    :param numImgs: number of images to be used (default is 3)
    """
    print ("Finding image correspondences")

    # Initialize Projection
    projection = ProjectionHelper(intrinsic, opt.depth_min, opt.depth_max, proj_image_dims, opt.voxel_size)

    # Find all scene chunks that have been precomputed
    fileList = list()
    for file in os.listdir(chunksPath):
        if file.endswith(".npy"):
            scene = file[:-4]
            fileList.append(scene)
    count = 1

    # Iterate through all scene chunks and compute their corresponding images
    for scene in fileList:
        if(os.path.isfile(os.path.join(outPath, scene + ".hdf5"))):
            print(scene + " was already processed.")
            continue
        poseDict = {}
        count += 1
        if(count % 50 == 0):
            print(count, "/", len(fileList))

        # Find the name of the scene (we need this to find the corresponding camera poses)
        countLiterals = 0
        for i in range(len(scene)):
            if(scene[i]=="0"):
                countLiterals = i
                break
        scene_nr = int(scene[countLiterals:countLiterals+4])
        scene_version = int(scene[countLiterals+5:countLiterals+7])

        # Load data
        data = np.load(os.path.join(chunksPath,  scene + ".npy"))
        scene_points = data[:, :3]
        semantic_labels = data[:, 3]
        npoints = scene_points.shape[0]

        # Find full Scene Number (always starts with '0')
        findZero = 0
        for i in range(len(scene)):
            if(scene[i] == '0'):
                findZero = i
                break
        large_scene = scene[findZero:findZero+7] # name of whole scene, e.g. 0000_01
        posesPathScene = os.path.join(posesPath, "scene"+large_scene, "pose")

        # Check if there are image poses for this scene (a couple of scenes caused problems when extracting the poses from sensor data)
        if(not os.path.isdir(posesPathScene)):
            print("Did not find any according Image Poses")
            continue

        # Iterate through all poses
        # For each pose, compute the number of points that lie in the frustum that corresponds to the camera pose
        # Keep the 3 image IDs corresponding to the poses with the highest numbers of points in the frustum
        for poseFile in os.listdir(posesPathScene):
            pose = load_pose(os.path.join(posesPathScene,poseFile))
            corners = projection.compute_frustum_corners(pose)[:, :3, 0] # Corners of Frustum
            normals = projection.compute_frustum_normals(corners) # Normals of frustum
            num_valid_points = projection.points_in_frustum(corners.double().cuda(), normals.double().cuda(), torch.DoubleTensor(scene_points).cuda()) # Checks for each point if it lies on the correct side of the normals of the frustum
            poseDict[poseFile[:-4]] = num_valid_points
        if(len(poseDict) == 0): # If there was something wrong, skip
            continue
        poseList = list()
        poseList.append(scene_nr)
        poseList.append(scene_version)
        for i in range(numImgs): # find maxima
            maximum = max(poseDict, key=poseDict.get)
            poseList.append(int(maximum))
            del poseDict[maximum]

        # Write to file
        h5file = h5py.File(os.path.join(outPath, scene + ".hdf5"), "w")
        dset = h5file.create_dataset("points", (npoints, 3), data=scene_points)
        dset = h5file.create_dataset("labels", (npoints,), data=semantic_labels)
        dset = h5file.create_dataset("corresponding_images", (numImgs+2,), data=poseList)
        h5file.close()




def combine_to_final_data_Split(in_path, out_path, num_scenes_per_file=1000, with_images=True):
    """
    Puts the single scene chunks into large containers with 1000 scene chunks each (randomly shuffled)
    :param in_path: path where scene chunk files (containing corresponding images) are stored
    :param out_path: where the output containers should be stored
    :param num_scenes_per_file:
    :param with_images: if false: creates containers without image correspondences (can be used for pointnet preprocessing)
    """

    # Create file lists
    fileListTrain = list()
    fileListVal = list()
    fileListTest = list()
    for file in os.listdir(in_path):
        if file.endswith(".hdf5"):
            scene = file
            if (scene.startswith("train")):
                fileListTrain.append(scene)
            if (scene.startswith("test")):
                fileListTest.append(scene)
            if(scene.startswith("val")):
                fileListVal.append(scene)

    # Create containers for each split
    for split in ["train", "test", "val"]:
        file_count = 0
        scene_count = 0
        if (split == "train"):
            fileList = fileListTrain
        if (split == "val"):
            fileList = fileListVal
        if (split == "test"):
            fileList = fileListTest
        dataDict = {}
        labelDict = {}
        if(with_images):
            imagesDict = {}

        # Iterate through all scene chunks and combine them to containers of 1000 chunks each
        for i in range(len(fileList)):
            if(i % 50 == 0):
                print(split, i, "/", len(fileList))
            scene_count += 1
            in_file = h5py.File(os.path.join(in_path, fileList[i]), "r")
            in_data = np.expand_dims(in_file['points'], 0)
            in_labels = np.expand_dims(in_file['labels'], 0)
            dataDict[i] = in_data
            labelDict[i] = in_labels
            if (with_images):
                in_images = np.expand_dims(np.asarray(in_file['corresponding_images']), 0)
                imagesDict[i] = in_images
            in_file.close()
            if (scene_count == num_scenes_per_file or i == len(fileList)-1):
                print("Processed container number " + str(file_count) + " for split " + split)
                out_file = h5py.File(os.path.join(out_path, split + "scene_container_" + str(file_count)) + ".hdf5",'w')
                pointGrp = out_file.create_group('points')
                labelsGrp = out_file.create_group('labels')
                for k, v in dataDict.items():
                    pointGrp.create_dataset(str(k), data=v)
                for k, v in labelDict.items():
                    labelsGrp.create_dataset(str(k), data=v)

                if(with_images):
                    imagesGrp = out_file.create_group('corresponding_images')
                    for k, v in imagesDict.items():
                        imagesGrp.create_dataset(str(k), data=v)
                scene_count = 0
                file_count += 1
                out_file.close()
                dataDict = {}
                labelDict = {}
                if (with_images):
                    imagesDict = {}



"""
Process numpy scenes to final scene chunk containers
"""

rawScenesPath = "/media/lorenzlamm/My Book/pointnet2/scannet/preprocessing/scannet_scenes"
path_to_benchmark = '/media/lorenzlamm/My Book/Final_Scannet_Data'
chunkPath = '/media/lorenzlamm/My Book/Final_Scannet_Data/sceneChunks'
chunks_With_Corr = '/media/lorenzlamm/My Book/Final_Scannet_Data/sceneChunks_With_Corr'
img_path = '/media/lorenzlamm/My Book/Scannet/out_images'
final_out_path = '/media/lorenzlamm/My Book/Final_Scannet_Data/final_containers'

findPointCloudsSplit(path_to_benchmark,rawScenesPath,chunkPath)
findCorrespondingImages(chunkPath, img_path, chunks_With_Corr)
combine_to_final_data_Split(chunks_With_Corr, final_out_path)

"""
Process an overfit sample of a single scene
"""

rawScenesPath = "/home/lorenzlamm/Dokumente/sampleBeachData/pointclouds"
path_to_benchmark = '/home/lorenzlamm/Dokumente/sampleBeachData'
chunkPath = '/home/lorenzlamm/Dokumente/sampleBeachData/scene_chunks_new'
chunks_With_Corr = '/home/lorenzlamm/Dokumente/sampleBeachData/scene_chunks_with_corr_new'
img_path = '/home/lorenzlamm/Dokumente/sampleBeachData/2d_data'
final_out_path = '/home/lorenzlamm/Dokumente/sampleBeachData/final_containers_new'

findPointCloudsSplit(path_to_benchmark,rawScenesPath,chunkPath)
findCorrespondingImages(chunkPath, img_path, chunks_With_Corr)
combine_to_final_data_Split(chunks_With_Corr, final_out_path)

