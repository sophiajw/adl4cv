from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
import etw_pytorch_utils as pt_utils
import pprint
import os.path as osp
import os
import argparse
import torch
import data_util


from pointnet2.models import Pointnet2SemMSG as Pointnet
from pointnet2.models.pointnet2_msg_sem import model_fn_decorator
from pointnet2.data import Indoor3DSemSeg

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-batch_size", type=int, default=1, help="Batch size [default: 32]"
)
parser.add_argument(
    "-num_points",
    type=int,
    default=4096,
    help="Number of points to train with [default: 4096]",
)
parser.add_argument(
    "-weight_decay",
    type=float,
    default=0,
    help="L2 regularization coeff [default: 0.0]",
)
parser.add_argument(
    "-lr", type=float, default=1e-2, help="Initial learning rate [default: 1e-2]"
)
parser.add_argument(
    "-lr_decay",
    type=float,
    default=0.5,
    help="Learning rate decay gamma [default: 0.5]",
)
parser.add_argument(
    "-decay_step",
    type=float,
    default=2e5,
    help="Learning rate decay step [default: 20]",
)
parser.add_argument(
    "-bn_momentum",
    type=float,
    default=0.9,
    help="Initial batch norm momentum [default: 0.9]",
)
parser.add_argument(
    "-bn_decay",
    type=float,
    default=0.5,
    help="Batch norm momentum decay gamma [default: 0.5]",
)
parser.add_argument(
    "-checkpoint", type=str, default=None, help="Checkpoint to start from"
)
parser.add_argument(
    "-epochs", type=int, default=200, help="Number of epochs to train for"
)
parser.add_argument(
    "-run_name",
    type=str,
    default="sem_seg_run_1",
    help="Name for run in tensorboard_logger",
)
parser.add_argument("--visdom-port", type=int, default=8097)
parser.add_argument("--visdom", action="store_true")

lr_clip = 1e-5
bnm_clip = 1e-2

args = parser.parse_args()

test_set = Indoor3DSemSeg(args.num_points, train=False)
test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
)

train_set = Indoor3DSemSeg(args.num_points)
train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    pin_memory=True,
    num_workers=2,
    shuffle=True,
)


model = torch.load("modelLog110.model")
print("JI")
points, labels, frames = data_util.load_hdf5_data("/home/lorenzlamm/Dokumente/final_network/scene_container_0.hdf5", 21)
print(points.shape)
points = points[0].unsqueeze(0)
print("HI")
out = model(points.cuda())
print(out.shape)
print(out)
print(torch.argmax(out, 2).shape)
pred_labels = torch.argmax(out, 2)
print(pred_labels)
print(torch.unique(pred_labels))
labels = labels[0].unsqueeze(0)
print(torch.unique(labels))
print(torch.sum(labels!=pred_labels.cpu()))

for 