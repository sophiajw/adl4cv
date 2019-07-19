
import argparse
import os, sys, inspect, time
import random
import torch
import importlib
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
import torch.optim.lr_scheduler as lr_sched
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
import itertools

import util
import data_util
from model import Model2d3d
from model import model_fn_decorator
from enet import create_enet_for_3d
from projection import ProjectionHelper
from data.Indoor3DSemSegLoader import Indoor3DSemSeg


sys.path.append(".")
from lib.solver import Solver
from lib.dataset import ScannetDataset, ScannetDatasetWholeScene, collate_random, collate_wholescene
from lib.loss import WeightedCrossEntropyLoss
from lib.config import CONF


ENET_TYPES = {'scannet': (41, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}  #classes, color mean/std 

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--train_data_list', required=False, default='/media/lorenzlamm/My Book/processing/final_training_files/hdf5_files.txt', help='path to file list of h5 train data')
parser.add_argument('--input_folder_3d', required=False, default='/workspace/beachnet_train/bn_train_data')
parser.add_argument('--val_data_list', default='', help='path to file list of h5 val data')
parser.add_argument('--output', default='./logs', help='folder to output model checkpoints')
parser.add_argument('--data_path_2d', required=False, default='/media/lorenzlamm/My Book/Scannet/out_images', help='path to 2d train data')
parser.add_argument('--class_weight_file', default='', help='path to histogram over classes')
# train params
parser.add_argument('--num_classes', default=42, help='#classes')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--max_epoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--lr_pointnet', type=float, default=1e-3, help='Initial learning rate for PointNet [default: 1e-2]')
parser.add_argument("--lr_decay", type=float, default=0.5, help="Learning rate decay gamma [default: 0.5]")
parser.add_argument("--lr_decay_pn", type=float, default=0.7, help="Learning rate decay [0.7 from pointnet++ paper]")
parser.add_argument("--decay_step", type=float, default=2e5, help="Learning rate decay step [default: 20]")
parser.add_argument("--bn_momentum", type=float, default=0.9, help="Initial batch norm momentum [default: 0.9")
parser.add_argument("--bn_decay", type=float, default=0.5, help="Batch norm momentum decay gamma [default: 0.5]")
parser.add_argument('--momentum', type=float, default=0.9, help='momentum, default=0.9')
parser.add_argument('--num_nearest_images', type=int, default=3, help='#images')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay, default=0.0005')
parser.add_argument('--weight_decay_pointnet', type=float, default=0, help='L2 regularization coeff [default: 0.0]')
parser.add_argument('--retrain', default='', help='model to load')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--model2d_type', default='scannet', help='which enet (scannet)')
parser.add_argument('--model2d_path', required=False, default='/workspace/beachnet_train/bn_train_data/scannetv2_enet.pth', help='path to enet model')
parser.add_argument('--use_proxy_loss', dest='use_proxy_loss', action='store_true')
parser.add_argument('--num_points', default=4096, help='number of points in one sample')
# 2d/3d 
parser.add_argument('--accuracy', type=float, default=0.05, help='accuracy of point projection (in meters)')
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
assert opt.model2d_type in ENET_TYPES
print(opt)

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader, stamp, weight, is_wholescene):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pointnet2/'))
    Pointnet = importlib.import_module("pointnet2_msg_semseg")

    model = Pointnet.get_model(num_classes=21).cuda()
    num_params = get_num_params(model)
    criterion = WeightedCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr_pointnet, weight_decay=opt.weight_decay)
    solver = Solver(model, dataloader, criterion, optimizer, opt.batch_size, stamp, is_wholescene)

    return solver, num_params

def save_info(args, root, train_examples, val_examples, num_params):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = train_examples
    info["num_val"] = val_examples
    info["num_params"] = num_params

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(opt.gpu)

# create camera intrinsics
input_image_dims = [328, 256]
proj_image_dims = [41, 32]
intrinsic = util.make_intrinsic(opt.fx, opt.fy, opt.mx, opt.my)
intrinsic = util.adjust_intrinsic(intrinsic, [opt.intrinsic_image_width, opt.intrinsic_image_height], proj_image_dims)
intrinsic = intrinsic.cuda()
grid_dims = [opt.grid_dimX, opt.grid_dimY, opt.grid_dimZ]
column_height = opt.grid_dimZ
batch_size = opt.batch_size
num_images = opt.num_nearest_images
grid_centerX = opt.grid_dimX // 2
grid_centerY = opt.grid_dimY // 2
color_mean = ENET_TYPES[opt.model2d_type][1]
color_std = ENET_TYPES[opt.model2d_type][2]
input_channels = 128
num_points = opt.num_points

# create enet and pointnet++ models
num_classes = opt.num_classes
model2d_fixed, model2d_trainable, model2d_classifier = create_enet_for_3d(ENET_TYPES[opt.model2d_type], opt.model2d_path, num_classes)
model = Model2d3d(num_classes, num_images, input_channels, intrinsic, proj_image_dims, opt.depth_min, opt.depth_max, opt.accuracy)
projection = ProjectionHelper(intrinsic, opt.depth_min, opt.depth_max, proj_image_dims, opt.accuracy)
# create loss
criterion_weights = torch.ones(num_classes) 
if opt.class_weight_file:
    criterion_weights = util.read_class_weights_from_file(opt.class_weight_file, num_classes, True)
for c in range(num_classes):
    if criterion_weights[c] > 0:
        criterion_weights[c] = 1 / np.log(1.2 + criterion_weights[c])
print(criterion_weights.numpy())
#raw_input('')
criterion = WeightedCrossEntropyLoss()
#criterion = torch.nn.CrossEntropyLoss(criterion_weights).cuda()
criterion2d = torch.nn.CrossEntropyLoss(criterion_weights).cuda()

# move to gpu
model2d_fixed = model2d_fixed.cuda()
model2d_fixed.eval()
model2d_trainable = model2d_trainable.cuda()
model2d_classifier = model2d_classifier.cuda()
model = model.cuda()
criterion = criterion.cuda()

# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr_pointnet, weight_decay=opt.weight_decay_pointnet)
optimizer2d = torch.optim.SGD(model2d_trainable.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
if opt.use_proxy_loss:
    optimizer2dc = torch.optim.SGD(model2d_classifier.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

# function to compute accuracy and miou in pointnet++
model_fn = model_fn_decorator(nn.CrossEntropyLoss())

# load data
# pointnet++
# if opt.wholescene:
#     is_wholescene = True
# else:
is_wholescene = False
train_dataset = Indoor3DSemSeg(num_points, root=opt.input_folder_3d, train=True)
val_dataset = Indoor3DSemSeg(num_points, root=opt.input_folder_3d, train=False)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=2
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    pin_memory=True,
    num_workers=2,
    shuffle=True
)
dataloader = {
    "train": train_dataloader,
    "val": val_dataloader
}
weight = train_dataset.labelweights
train_examples = len(train_dataset)
val_examples = len(val_dataset)

print("initializing...")
stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
root = os.path.join(CONF.OUTPUT_ROOT, stamp)
os.makedirs(root, exist_ok=True)
solver, num_params = get_solver(opt, dataloader, stamp, weight, is_wholescene)

# train_files = util.read_lines_from_file(opt.train_data_list)
# val_files = [] if not opt.val_data_list else util.read_lines_from_file(opt.val_data_list)
# print('#train files = ', len(train_files))
# print('#val files = ', len(val_files))

_SPLITTER = ','
confusion = tnt.meter.ConfusionMeter(num_classes)
confusion2d = tnt.meter.ConfusionMeter(num_classes)
confusion_val = tnt.meter.ConfusionMeter(num_classes)
confusion2d_val = tnt.meter.ConfusionMeter(num_classes)

print("\n[info]")
print("Train examples: {}".format(train_examples))
print("Evaluation examples: {}".format(val_examples))
print("Start training...\n")
save_info(opt, root, train_examples, val_examples, num_params)

def train(epoch, iter, log_file, train_file, log_file_2d):
    train_loss = []

    if opt.use_proxy_loss:
        model2d_classifier.train()

    #points, labels, frames = data_util.load_hdf5_data(train_file, num_classes)
    #num_points = points.shape[1]
    # shape of points: (1000, 8192, 3)
    # shape of labels: (1000, 8192,)
    # shape of frames: (1000, 5,): 0th entry: scene (0000), 1st entry: version of scene(00), 2-4th entry: image ids (for 3 images)
    #frames = frames[:, :2+num_images]
    # volumes = volumes.permute(0, 1, 4, 3, 2)
    # no permutation necessary
    train_loss_2d = []
    model.train()
    start = time.time()
    model2d_trainable.train()
    # labels = labels.permute(0, 1, 4, 3, 2)
    # no permutation necessary

    # labels = labels[:, 0, :, grid_centerX, grid_centerY]  # center columns as targets
    # not necessary; want to predict every point in cloud

    #num_samples = points.shape[0]
    # TODO: concatenate all hdf5 files to one large file
    # shuffle
    #indices = torch.randperm(num_samples).long().split(batch_size)
    # remove last mini-batch so that all the batches have equal size
    #indices = indices[:-1]

    # initialize Tensors for depth, color, camera pose, labels for projection pass
    depth_images = torch.cuda.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
    color_images = torch.cuda.FloatTensor(batch_size * num_images, 3, input_image_dims[1], input_image_dims[0])
    camera_poses = torch.cuda.FloatTensor(batch_size * num_images, 4, 4)
    label_images = torch.cuda.LongTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])

    for t, data in enumerate(train_dataloader):
        points, targets, frames = data
        # targets = torch.autograd.Variable(labels[v].cuda())
        # valid targets
        mask = targets.view(-1).data.clone()
        for k in range(num_classes):
            if criterion_weights[k] == 0:
                mask[mask.eq(k)] = 0 # excludes all objects that are not contained in class list
        maskindices = mask.nonzero().squeeze()
        if len(maskindices.shape) == 0:
            continue

        data_util.load_frames_multi(opt.data_path_2d, frames, depth_images, color_images, camera_poses, color_mean, color_std)

        points_projection = torch.repeat_interleave(points, num_images, dim=0)
        # compute projection mapping
        proj_mapping = [projection.compute_projection(p, d, c, num_points) for p, d, c in zip(points_projection, depth_images, camera_poses)]
        if None in proj_mapping: # invalid sample
            # print '(invalid sample)'
            continue
        proj_mapping = list(zip(*proj_mapping))
        proj_ind_3d = torch.stack(proj_mapping[0])
        proj_ind_2d = torch.stack(proj_mapping[1])

        if opt.use_proxy_loss:
            data_util.load_label_frames(opt.data_path_2d, frames, label_images, num_classes)
            mask2d = label_images.view(-1).clone()
            for k in range(num_classes):
                if criterion_weights[k] == 0:
                    mask2d[mask2d.eq(k)] = 0
            mask2d = mask2d.nonzero().squeeze()
            if (len(mask2d.shape) == 0):
                continue  # nothing to optimize for here
        # 2d
        imageft_fixed = model2d_fixed(torch.autograd.Variable(color_images))
        imageft = model2d_trainable(imageft_fixed)
        if opt.use_proxy_loss:
            ft2d = model2d_classifier(imageft)
            ft2d = ft2d.permute(0, 2, 3, 1).contiguous()

        # 2d/3d
        input3d = torch.autograd.Variable(points.cuda())
        output = model(input3d, imageft, torch.autograd.Variable(proj_ind_3d), torch.autograd.Variable(proj_ind_2d))

        pred = output
        num_classes = pred.size(2)
        loss = criterion(pred.contiguous().view(-1, num_classes), targets.view(-1), weight.view(-1))

        # loss = criterion(output.view(-1, num_classes), targets.view(-1))
        # _, loss, _ = model_fn(model, (points, targets), imageft, torch.autograd.Variable(proj_ind_3d), torch.autograd.Variable(proj_ind_2d))

        train_loss.append(loss.item())
        optimizer.zero_grad()
        optimizer2d.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer2d.step()
        if opt.use_proxy_loss:
            loss2d = criterion2d(ft2d.view(-1, num_classes), torch.autograd.Variable(label_images.view(-1)))
            train_loss_2d.append(loss2d.item())
            optimizer2d.zero_grad()
            optimizer2dc.zero_grad()
            loss2d.backward()
            optimizer2dc.step()
            optimizer2d.step()
            # confusion
            y = ft2d.data
            y = y.view(-1, num_classes)[:, :-1]
            _, predictions = y.max(1)
            predictions = predictions.view(-1)
            k = label_images.view(-1)
            confusion2d.add(torch.index_select(predictions, 0, mask2d), torch.index_select(k, 0, mask2d))

        # confusion
        y = output.data
        y = y.view(y.nelement()/y.size(2), num_classes)[:, :-1]
        _, predictions = y.max(1)
        predictions = predictions.view(-1)
        k = targets.data.view(-1)
        # computes the confustion matrix
        confusion.add(torch.index_select(predictions, 0, maskindices), torch.index_select(k, 0, maskindices))
        log_file.write(_SPLITTER.join([str(f) for f in [epoch, iter, loss.item()]]) + '\n')
        iter += 1
        if iter % 10000 == 0:
            torch.save(model.state_dict(), os.path.join(opt.output, 'model-iter%s-epoch%s.pth' % (iter, epoch)))
            torch.save(model2d_trainable.state_dict(), os.path.join(opt.output, 'model2d-iter%s-epoch%s.pth' % (iter, epoch)))
            if opt.use_proxy_loss:
                torch.save(model2d_classifier.state_dict(), os.path.join(opt.output, 'model2dc-iter%s-epoch%s.pth' % (iter, epoch)))
        if iter == 1:
            torch.save(model2d_fixed.state_dict(), os.path.join(opt.output, 'model2dfixed.pth'))

        if iter % 100 == 0:
            evaluate_confusion(confusion, train_loss, epoch, iter, -1, 'Train', log_file)
            if opt.use_proxy_loss:
                evaluate_confusion(confusion2d, train_loss_2d, epoch, iter, -1, 'Train2d', log_file_2d)

    end = time.time()
    took = end - start
    evaluate_confusion(confusion, train_loss, epoch, iter, took, 'Train', log_file)
    if opt.use_proxy_loss:
        evaluate_confusion(confusion2d, train_loss_2d, epoch, iter, took, 'Train2d', log_file_2d)
    return train_loss, iter, train_loss_2d


def test(epoch, iter, log_file, val_file, log_file_2d):
    test_loss = []
    test_loss_2d = []
    model.eval()
    model2d_fixed.eval()
    model2d_trainable.eval()
    if opt.use_proxy_loss:
        model2d_classifier.eval()
    start = time.time()

    points, labels, frames = data_util.load_hdf5_data(val_file, num_classes)
    num_points = points.shape[1]

    frames = frames[:, :2+num_images]
    num_samples = points.shape[0]
    # shuffle
    indices = torch.randperm(num_samples).long().split(batch_size)
    # remove last mini-batch so that all the batches have equal size
    indices = indices[:-1]

    with torch.no_grad():
        depth_images = torch.cuda.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
        color_images = torch.cuda.FloatTensor(batch_size * num_images, 3, input_image_dims[1], input_image_dims[0])
        camera_poses = torch.cuda.FloatTensor(batch_size * num_images, 4, 4)
        label_images = torch.cuda.LongTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])

        for t,v in enumerate(indices):
            targets = labels[v].cuda()
            # valid targets
            mask = targets.view(-1).data.clone()
            for k in range(num_classes):
                if criterion_weights[k] == 0:
                    mask[mask.eq(k)] = 0
            maskindices = mask.nonzero().squeeze()
            if len(maskindices.shape) == 0:
                continue

            # get 2d data
            data_util.load_frames_multi(opt.data_path_2d, frames[v], depth_images, color_images, camera_poses, color_mean, color_std)
            if opt.use_proxy_loss:
                data_util.load_label_frames(opt.data_path_2d, frames[v], label_images, num_classes)
                mask2d = label_images.view(-1).clone()
                for k in range(num_classes):
                    if criterion_weights[k] == 0:
                        mask2d[mask2d.eq(k)] = 0
                mask2d = mask2d.nonzero().squeeze()
                if (len(mask2d.shape) == 0):
                    continue  # nothing to optimize for here

            # compute projection mapping
            proj_mapping = [projection.compute_projection(p, d, c, num_points) for p, d, c in zip(points[v], depth_images, camera_poses)]
            if None in proj_mapping: #invalid sample
                #print '(invalid sample)'
                continue
            proj_mapping = list(zip(*proj_mapping))
            proj_ind_3d = torch.stack(proj_mapping[0])
            proj_ind_2d = torch.stack(proj_mapping[1])
            # 2d
            imageft_fixed = model2d_fixed(color_images)
            imageft = model2d_trainable(imageft_fixed)
            if opt.use_proxy_loss:
                ft2d = model2d_classifier(imageft)
                ft2d = ft2d.permute(0, 2, 3, 1).contiguous()
            # 2d/3d
            input3d = points[v].cuda()
            output = model(input3d, imageft, proj_ind_3d, proj_ind_2d)
            loss = criterion(output.view(-1, num_classes), targets.view(-1))
            test_loss.append(loss.item())
            if opt.use_proxy_loss:
                loss2d = criterion2d(ft2d.view(-1, num_classes), label_images.view(-1))
                test_loss_2d.append(loss2d.item())
                # confusion
                y = ft2d.data
                y = y.view(-1, num_classes)[:, :-1]
                _, predictions = y.max(1)
                predictions = predictions.view(-1)
                k = label_images.view(-1)
                confusion2d_val.add(torch.index_select(predictions, 0, mask2d), torch.index_select(k, 0, mask2d))
            
            # confusion
            y = output.data
            y = y.view(y.nelement()/y.size(2), num_classes)[:, :-1]
            _, predictions = y.max(1)
            predictions = predictions.view(-1)
            k = targets.data.view(-1)
            confusion_val.add(torch.index_select(predictions, 0, maskindices), torch.index_select(k, 0, maskindices))

    end = time.time()
    took = end - start
    evaluate_confusion(confusion_val, test_loss, epoch, iter, took, 'Test', log_file)
    if opt.use_proxy_loss:
         evaluate_confusion(confusion2d_val, test_loss_2d, epoch, iter, took, 'Test2d', log_file_2d)
    return test_loss, test_loss_2d


def evaluate_confusion(confusion_matrix, loss, epoch, iter, time, which, log_file):
    conf = confusion_matrix.value()
    total_correct = 0
    valids = np.zeros(num_classes, dtype=np.float32)
    iou = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        num = conf[c, :].sum() # number of points with ground truth c (TP + TN)
        valids[c] = -1 if num == 0 else float(conf[c][c]) / float(num) # TP / (TP + TN)
        total_correct += conf[c][c]
        F = conf[:, c].sum() # number of points predicted to be in class c (FP + FN)
        iou[c] = 0 if conf[c, c]+F == 0 else float(conf[c, c]) / float(conf[c, c]+F) # TP / (TP + FP + FN)
    instance_acc = -1 if conf.sum() == 0 else float(total_correct) / float(conf.sum())
    avg_acc = -1 if np.all(np.equal(valids, -1)) else np.mean(valids[np.not_equal(valids, -1)])
    mean_iou = np.mean(iou)
    log_file.write(_SPLITTER.join([str(f) for f in [epoch, iter, torch.mean(torch.Tensor(loss)), avg_acc, instance_acc, mean_iou, time]]) + '\n')
    log_file.flush()

    print('{} Epoch: {}\tIter: {}\tLoss: {:.6f}\tAcc(inst): {:.6f}\tAcc(avg): {:.6f}\tmIoU: {:.6f}\tTook: {:.2f}'.format(
        which, epoch, iter, torch.mean(torch.Tensor(loss)), instance_acc, avg_acc, mean_iou, time))


def main():
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
    log_file = open(os.path.join(opt.output, 'log.csv'), 'w')
    log_file.write(_SPLITTER.join(['epoch','iter','loss','avg acc', 'instance acc', 'mIoU', 'time']) + '\n')
    log_file.flush()
    log_file_2d = None
    if opt.use_proxy_loss:
        log_file_2d = open(os.path.join(opt.output, 'log2d.csv'), 'w')
        log_file_2d.write(_SPLITTER.join(['epoch','iter','loss','avg acc', 'instance acc', 'time']) + '\n')
        log_file_2d.flush()

    has_val = len(val_dataset) > 0
    if has_val:
        log_file_val = open(os.path.join(opt.output, 'log_val.csv'), 'w')
        log_file_val.write(_SPLITTER.join(['epoch', 'iter', 'loss','avg acc', 'instance acc', 'mIoU', 'time']) + '\n')
        log_file_val.flush()
        log_file_2d_val = None
        if opt.use_proxy_loss:
            log_file_2d_val = open(os.path.join(opt.output, 'log2d_val.csv'), 'w')
            log_file_2d_val.write(_SPLITTER.join(['epoch','iter','loss','avg acc', 'instance acc', 'time']) + '\n')
            log_file_2d_val.flush()
    # start training
    print('starting training...')
    iter = 0
    num_files_per_val = 10
    for epoch in range(opt.max_epoch):
        train_loss = []
        train2d_loss = []
        val_loss = []
        val2d_loss = []

        # go thru shuffled train files
        #train_file_indices = torch.randperm(len(train_files))
        for k in range(len(train_dataloader)):
            print('Epoch: {}\tFile: {}/{}\t{}'.format(epoch, k, len(train_dataloader), train_dataloader))
            loss, iter, loss2d = train(epoch, iter, log_file, train_dataloader, log_file_2d)
            train_loss.extend(loss)
            if loss2d:
                 train2d_loss.extend(loss2d)
            if has_val and k % num_files_per_val == 0:
                # val_index = torch.randperm(len(val_dataloader))[0]
                loss, loss2d = test(epoch, iter, log_file_val, val_dataloader, log_file_2d_val)
                val_loss.extend(loss)
                if loss2d:
                     val2d_loss.extend(loss2d)
        evaluate_confusion(confusion, train_loss, epoch, iter, -1, 'Train', log_file)
        if opt.use_proxy_loss:
            evaluate_confusion(confusion2d, train2d_loss, epoch, iter, -1, 'Train2d', log_file_2d)
        if has_val:
            evaluate_confusion(confusion_val, val_loss, epoch, iter, -1, 'Test', log_file_val)
            if opt.use_proxy_loss:
                evaluate_confusion(confusion2d_val, val_loss, epoch, iter, -1, 'Test2d', log_file_2d_val)
        torch.save(model.state_dict(), os.path.join(opt.output, 'model-epoch-%s.pth' % epoch))
        torch.save(model2d_trainable.state_dict(), os.path.join(opt.output, 'model2d-epoch-%s.pth' % epoch))
        if opt.use_proxy_loss:
            torch.save(model2d_classifier.state_dict(), os.path.join(opt.output, 'model2dc-epoch-%s.pth' % epoch))
        confusion.reset()
        confusion2d.reset()
        confusion_val.reset()
        confusion2d_val.reset()
    log_file.close()
    if has_val:
        log_file_val.close()
    if opt.use_proxy_loss:
        log_file_2d.close()
        if has_val:
            log_file_2d_val.close()


if __name__ == '__main__':
    main()


