from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import time



def _get_data_files(list_filename):
    """
    reads the names of the scene files into a list and returns it
    """
    with open(list_filename) as f:
        return [line.rstrip() for line in f]


def _load_data_file(name):
    """
    Returns the data out of a hdf5 file
    :param name: name of scene
    :return: data: 3D point coordinates; group with each entry of shape [1, nr_of_points_in_scene, 3]
             label: labels for each point: group with each entry of shape [1, nr_of_points_in_scene]
             frames: corresponding images to each scene chunk: [1, 5]
    """
    f = h5py.File(name)
    data = f["points"]
    label = f["labels"]
    frames = f["corresponding_images"]
    return data, label, frames


class DataLoader(data.Dataset):
    def __init__(self, num_points, root, train=True, test=False, visualize = False, vis_scene = "0000_00"):
        """
        Initialization of Dataset
        :param num_points: points per scene chunk
        :param root: directory where to find the scene chunk containers
        :param train: True, if training dataset, False if validation dataset
        :param test:
        :param visualize:
        :param vis_scene:
        """

        super().__init__()
        BASE_DIR = root
        self.visualize = visualize
        self.vis_scene = vis_scene
        self.folder = "bn_train_data"
        self.data_dir = os.path.join(BASE_DIR, self.folder)

        self.train, self.num_points = train, num_points
        self.test = test
        if(self.test):
            self.train = False
        if(self.visualize):
            self.test = False

        # load the desired scenes according to given parameters
        if(self.train):
            with open(os.path.join(root, "all_files_train.txt"), 'w+') as f:
                list = os.listdir(root)
                for entry in list:
                    if(entry.startswith("train")):
                        f.writelines(os.path.join(root, entry + "\n"))
            all_files = _get_data_files(os.path.join(root, "all_files_train.txt"))
        elif(self.test):
            with open(os.path.join(root, "all_files_test.txt"), 'w+') as f:
                list = os.listdir(root)
                for entry in list:
                    if(entry.startswith("test")):
                        countTo5 += 1
                        f.writelines(os.path.join(root, entry + "\n"))
            all_files = _get_data_files(os.path.join(root, "all_files_test.txt"))
        elif(self.visualize):
            with open(os.path.join(root, "all_files_vis.txt"), 'w+') as f:
                list = os.listdir(root)
                for entry in list:
                    if(entry.startswith("test")):
                        f.writelines(os.path.join(root, entry + "\n"))
            all_files = _get_data_files(os.path.join(root, "all_files_vis.txt"))
        else:
            with open(os.path.join(root, "all_files_val.txt"), 'w+') as f:
                list = os.listdir(root)
                for entry in list:
                    if(entry.startswith("val")):
                        f.writelines(os.path.join(root,entry + "\n"))
            all_files = _get_data_files(os.path.join(root, "all_files_val.txt"))

        data_batchlist, label_batchlist, frames_batchlist = [], [], []
        count = 0

        if(self.visualize):
            for f in all_files:
                count += 1
                tempData, tempLabel, tempFrames = _load_data_file(f)
                largeScene = int(vis_scene[:4])
                scene_ID = int(vis_scene[5:7])
                for k, v in tempFrames.items():
                    if (v[:][0][0] == largeScene and v[:][0][1] == scene_ID):
                        v = tempData[k]
                        data_batchlist.append(v[:])
                        v = tempLabel[k]
                        label_batchlist.append(v[:])
                        v = tempFrames[k]
                        frames_batchlist.append(v[:])
        else:
            for f in all_files:
                count += 1
                print(count, "/", len(all_files))
                tempData, tempLabel, tempFrames = _load_data_file(f)
                for k, v in tempData.items():
                    data_batchlist.append(v[:])
                for k, v in tempLabel.items():
                    label_batchlist.append(v[:])
                for k, v in tempFrames.items():
                    frames_batchlist.append(v[:])

        self.points = data_batchlist
        self.labels = label_batchlist
        self.frames = frames_batchlist

        # Compute Label weights
        label_counts = np.ones(21)
        count_total_points = 1e-8
        for labels_it in label_batchlist:
            count_total_points += labels_it.shape[1]
            for i in range(21):
                label_counts[i] += np.sum(labels_it[0] == i)

        self.labelweights = label_counts / count_total_points
        for c in range(21):
            if (c == 0):
                self.labelweights[c] = 1.0
            else:
                self.labelweights[c] = 1 / np.log(1.2 + self.labelweights[c])




    def __getitem__(self, idx):
        # randomly samples num_points out of scene with index idx
        start = time.time()
        current_frames = torch.from_numpy(self.frames[idx][0])
        choice = np.random.choice(self.labels[idx][0].shape[0]-1, self.num_points, replace=True)
        current_points = torch.from_numpy(self.points[idx][0, choice].copy()).type(
            torch.FloatTensor
        )
        current_labels = torch.from_numpy(self.labels[idx][0, choice].copy()).type(
            torch.LongTensor
        )
        sample_weights = self.labelweights[current_labels]
        dummy = np.zeros((4096,0))
        fetch_time = time.time() - start
        return current_points, dummy, current_labels, current_frames, sample_weights, fetch_time

    def __len__(self):
        return int(len(self.points))

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass


if __name__ == "__main__":
    print("Main")
