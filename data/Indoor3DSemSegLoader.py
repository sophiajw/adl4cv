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
import subprocess
import shlex
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = "/content/beachnet_train"

def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip() for line in f]


def _load_data_file(name):
    f = h5py.File(name)
    data = f["points"]
    label = f["labels"]
    frames = f["corresponding_images"]
    return data, label, frames


class Indoor3DSemSeg(data.Dataset):
    def __init__(self, num_points, root, train=True, download=True, npoints=4096, data_precent=1.0, test=False):
        super().__init__()
        BASE_DIR = root
        self.npoints = npoints
        self.data_precent = data_precent
        self.folder = "bn_train_data"
        self.data_dir = os.path.join(BASE_DIR, self.folder)
        self.url = (
            "https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip"
        )

        download = False
        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.join(BASE_DIR, os.path.basename(self.url))
            subprocess.check_call(
                shlex.split("curl {} -o {}".format(self.url, zipfile))
            )

            subprocess.check_call(
                shlex.split("unzip {} -d {}".format(zipfile, BASE_DIR))
            )

            subprocess.check_call(shlex.split("rm {}".format(zipfile)))

        self.train, self.num_points = train, num_points
        self.test = test
        if(self.test):
            self.train=False

        if(self.train):
            with open(os.path.join(root, "all_files_train.txt"), 'w+') as f:
                list = os.listdir(root)
                for entry in list:
                    if(entry.startswith("train")):
                        f.writelines(os.path.join(root, entry + "\n"))
                        break
            all_files = _get_data_files(os.path.join(root, "all_files_train.txt"))
        elif(self.test):
            with open(os.path.join(root, "all_files_test.txt"), 'w+') as f:
                list = os.listdir(root)
                for entry in list:
                    if(entry.startswith("test")):
                        f.writelines(os.path.join(root, entry + "\n"))
                        break
            all_files = _get_data_files(os.path.join(root, "all_files_test.txt"))
        else:
            with open(os.path.join(root, "all_files_val.txt"), 'w+') as f:
                list = os.listdir(root)
                for entry in list:
                    if(entry.startswith("val")):
                        f.writelines(os.path.join(root,entry + "\n"))
                        break
            all_files = _get_data_files(os.path.join(root, "all_files_val.txt"))


        data_batchlist, label_batchlist, frames_batchlist = [], [], []
        count = 0
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


        label_counts = np.ones(21)
        count = 0
        count_total_points = 0
        print("Computing Labelweights")
        for labels_it in label_batchlist:
            count+=1
            print("counted", count, "/", len(label_batchlist), "point clouds")
            count_total_points += labels_it.shape[1]
            for i in range(21):
                label_counts[i] += np.sum(labels_it[0] == i)
        print(label_counts)

#        labels_unique = np.unique(labels_batches)
 #       labels_unique_count = np.stack([(labels_batches == labels_u).sum() for labels_u in labels_unique])

        self.labelweights = label_counts / count_total_points
        print(self.labelweights)
        for c in range(21):
            if (c == 0):
                self.labelweights[c] = 1.0
            else:
                self.labelweights[c] = 1 / np.log(1.2 + self.labelweights[c])

        self.points = data_batchlist
        self.labels = label_batchlist
        self.frames = frames_batchlist


    def __getitem__(self, idx):
        start = time.time()
        current_frames = torch.from_numpy(self.frames[idx][0])
        choice = np.random.choice(self.labels[idx][0].shape[0]-1, self.npoints, replace=True)
        current_points = torch.from_numpy(self.points[idx][0, choice].copy()).type(
            torch.FloatTensor
        )
        current_labels = torch.from_numpy(self.labels[idx][0, choice].copy()).type(
            torch.LongTensor
        )
        sample_weights = self.labelweights[current_labels]
        test = np.zeros((4096,0))
        fetch_time = time.time() - start
        return current_points, test, current_labels, current_frames, sample_weights, fetch_time

    def __len__(self):
        return int(len(self.points) * self.data_precent)

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass


if __name__ == "__main__":
    dset = Indoor3DSemSeg(4096, '/workspace/beachnet_train/bn_train_data', train=True)
    print(dset[0])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    for i, data in enumerate(dloader, 0):
        inputs, test, labels, frames, sample_weights, fetch_time = data
        if i == len(dloader) - 1:
            print(inputs.size())
