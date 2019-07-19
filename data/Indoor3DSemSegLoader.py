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
    data = f["points"][:]
    label = f["labels"][:]
    frames = f["corresponding_images"][:]
    return data, label, frames


class Indoor3DSemSeg(data.Dataset):
    def __init__(self, num_points, root, train=True, download=True, data_precent=1.0):
        super().__init__()
        BASE_DIR = root
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

        if(self.train):
            with open(os.path.join(root, "all_files.txt"), 'w+') as f:
                list = os.listdir(root)
                for entry in list:
                    if(entry.startswith("train")):
                        f.writelines(os.path.join(root, entry + "\n"))
            all_files = _get_data_files(os.path.join(root, "all_files.txt"))
        else:
            with open(os.path.join(root, "all_files.txt"), 'w+') as f:
                list = os.listdir(root)
                for entry in list:
                    if(entry.startswith("val")):
                        f.writelines(os.path.join(root,entry + "\n"))
            all_files = _get_data_files(os.path.join(root, "all_files.txt"))

        data_batchlist, label_batchlist, frames_batchlist = [], [], []
        for f in all_files:
            data, label, frames = _load_data_file(f)
            if(len(data.shape) == 2):
                data = np.expand_dims(data, axis=0)
                label = np.expand_dims(label, axis=0)
                frames = np.expand_dims(frames, axis=0)
            data_batchlist.append(data)
            label_batchlist.append(label)
            frames_batchlist.append(frames)



        data_batches = np.concatenate(data_batchlist, 0)
        labels_batches = np.concatenate(label_batchlist, 0)
        frames_batches = np.concatenate(frames_batchlist, 0)



        labels_unique = np.unique(labels_batches)
        labels_unique_count = np.stack([(labels_batches == labels_u).sum() for labels_u in labels_unique])

        labelSum = labels_unique_count.sum()
        self.labelweights = np.zeros(21)
        for c in range(21):
            if (c in labels_unique):
                count = 0
                for k in range(21):
                    if (c == k):
                        self.labelweights[count] = labels_unique_count[count] / labelSum
                    if (k in labels_unique):
                        count += 1
            else:
                self.labelweights[c] = 1
        # self.labelweights = labels_unique_count / (labels_unique_count.sum())
        for c in range(21):
            if (c == 0):
                self.labelweights[c] = 1.0
            else:
                if (c in labels_unique):
                    self.labelweights[c] = 1 / np.log(1.2 + self.labelweights[c])
                else:
                    self.labelweights[c] = 1.0





        test_area = "Area_5"
        train_idxs, test_idxs = [], []
        
        for i in range(data_batches.shape[0]):
            if(self.train):
                train_idxs.append(i)
            else:
                test_idxs.append(i)
        
        #for i, room_name in enumerate(room_filelist):
        #    if test_area in room_name:
        #        test_idxs.append(i)
        #    else:
        #        train_idxs.append(i)

        if self.train:
            self.points = data_batches[train_idxs, ...]
            self.labels = labels_batches[train_idxs, ...]
            self.frames = frames_batches[train_idxs, ...]
        else:
            self.points = data_batches[test_idxs, ...]
            self.labels = labels_batches[test_idxs, ...]
            self.frames = frames_batches[test_idxs, ...]

    def __getitem__(self, idx):
        start = time.time()
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)
        current_points = torch.from_numpy(self.points[idx, pt_idxs].copy()).type(
            torch.FloatTensor
        )
        current_labels = torch.from_numpy(self.labels[idx, pt_idxs].copy()).type(
            torch.LongTensor
        )
        current_frames = torch.from_numpy(self.frames[idx])
        fetch_time = time.time() - start


        sample_weights = self.labelweights[current_labels]
        test = np.zeros((4096,0))
        return current_points, test, current_labels, current_frames, sample_weights, fetch_time

    def __len__(self):
        return int(self.points.shape[0] * self.data_precent)

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
