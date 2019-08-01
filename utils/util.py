
import os, struct, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# util for saving tensors, for debug purposes
def write_array_to_file(tensor, filename):
    sz = tensor.shape
    with open(filename, 'wb') as f:
        f.write(struct.pack('Q', sz[0]))
        f.write(struct.pack('Q', sz[1]))
        f.write(struct.pack('Q', sz[2]))
        tensor.tofile(f)


def read_lines_from_file(filename):
    assert os.path.isfile(filename)
    lines = open(filename).read().splitlines()
    return lines


# create camera intrinsics
def make_intrinsic(fx, fy, mx, my):
    intrinsic = torch.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    # affine transformation from image plane to pixel coords
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0,0] *= float(resize_width)/float(intrinsic_image_dim[0])
    intrinsic[1,1] *= float(image_dim[1])/float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0,2] *= float(image_dim[0]-1)/float(intrinsic_image_dim[0]-1)
    intrinsic[1,2] *= float(image_dim[1]-1)/float(intrinsic_image_dim[1]-1)
    return intrinsic


def get_sample_files(samples_path):
    files = [f for f in os.listdir(samples_path) if f.endswith('.sample')] #and os.path.isfile(join(samples_path, f))]
    return files


def get_sample_files_for_scene(scene, samples_path):
    files = [f for f in os.listdir(samples_path) if f.startswith(scene) and f.endswith('.sample')] #and os.path.isfile(join(samples_path, f))]
    print('found ', len(files), ' for ', os.path.join(samples_path, scene))
    return files


def load_pose(filename):
    assert os.path.isfile(filename)
    pose = torch.Tensor(4, 4)
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
    return torch.from_numpy(np.asarray(lines).astype(np.float32))


def read_class_weights_from_file(filename, num_classes, normalize):
    assert os.path.isfile(filename)
    weights = torch.zeros(num_classes)
    lines = open(filename).read().splitlines()
    for line in lines:
        parts = line.split('\t')
        assert len(parts) == 2
        weights[int(parts[0])] = int(parts[1])
    if normalize:
        weights = weights / torch.sum(weights)
    return weights

def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model).__name__)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, weights):
        assert inputs.size(0) == targets.size(0) == weights.size(0)

        loss = F.cross_entropy(input=inputs, target=targets, reduction="none", ignore_index=self.ignore_index)
        loss = torch.mean(loss * weights.float())

        return loss
