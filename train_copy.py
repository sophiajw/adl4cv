
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

from tensorboardX import SummaryWriter
sys.path.append(".")
from lib.solver import Solver
from lib.dataset import ScannetDataset, ScannetDatasetWholeScene, collate_random, collate_wholescene
from lib.loss import WeightedCrossEntropyLoss
from lib.config import CONF

log = {phase: {} for phase in ["train", "val"]}
ENET_TYPES = {'scannet': (41, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}  #classes, color mean/std 
global_iter_id = 0
total_iter = {}


ITER_REPORT_TEMPLATE = """
----------------------iter: [{global_iter_id}/{total_iter}]----------------------
[loss] train_loss: {train_loss}
[sco.] train_acc: {train_acc}
[sco.] train_miou: {train_miou}
[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_iter_time: {mean_iter_time}s
"""

EPOCH_REPORT_TEMPLATE = """
------------------------summary------------------------
[train] train_loss: {train_loss}
[train] train_acc: {train_acc}
[train] train_miou: {train_miou}
[val]   val_loss: {val_loss}
[val]   val_acc: {val_acc}
[val]   val_miou: {val_miou}
"""

BEST_REPORT_TEMPLATE = """
-----------------------------best-----------------------------
[best] epoch: {epoch}
[loss] loss: {loss}
[sco.] acc: {acc}
[sco.] miou: {miou}
"""

iter_report_template = ITER_REPORT_TEMPLATE
epoch_report_template = EPOCH_REPORT_TEMPLATE
best_report_template = BEST_REPORT_TEMPLATE




# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--train_data_list', required=False, default='/media/lorenzlamm/My Book/processing/final_training_files/hdf5_files.txt', help='path to file list of h5 train data')
#parser.add_argument('--input_folder_3d', required=False, default='/workspace/beachnet_train/bn_train_data')
parser.add_argument('--input_folder_3d', required=False, default='/home/lorenzlamm/Dokumente/sampleBeachData/finalContainers')

parser.add_argument('--val_data_list', default='', help='path to file list of h5 val data')
parser.add_argument('--output', default='./logs', help='folder to output model checkpoints')
#parser.add_argument('--data_path_2d', required=False, default='/workspace/beachnet_train/bn_train_data', help='path to 2d train data')
parser.add_argument('--data_path_2d', required=False, default='/home/lorenzlamm/Dokumente/sampleBeachData/2d_data', help='path to 2d train data')

parser.add_argument('--class_weight_file', default='', help='path to histogram over classes')
# train params
parser.add_argument('--num_classes', default=21, help='#classes')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--max_epoch', type=int, default=50, help='number of epochs to train for')
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
#parser.add_argument('--model2d_path', required=False, default='/workspace/beachnet_train/bn_train_data/scannetv2_enet.pth', help='path to enet model')
parser.add_argument('--model2d_path', required=False, default='//home/lorenzlamm/Dokumente/final_new/adl4cv/scannetv2_enet.pth', help='path to enet model')
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
input_channels = 0
num_points = opt.num_points

# create enet and pointnet++ models
num_classes = opt.num_classes
model2d_fixed, model2d_trainable, model2d_classifier = create_enet_for_3d(ENET_TYPES[opt.model2d_type], opt.model2d_path, num_classes)
model = Model2d3d(num_classes, num_images, input_channels, intrinsic, proj_image_dims, opt.depth_min, opt.depth_max, opt.accuracy, fusion=False, fuse_no_ft_pn=True)
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

#all_frames = np.zeros((1000,1))
#for i in range(len(train_dataset)):
#    all_frames[i] = train_dataset[i][3][0]
#all_frames = all_frames.flatten()
#print(all_frames.shape)
#print(np.unique(all_frames))


val_dataloader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=True,
    pin_memory=True,
    num_workers=8
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    pin_memory=True,
    num_workers=8,
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

def train(epoch, iter, log_file, train_dataloader, log_file_2d):
    global global_iter_id
    ## Parameters for logging
    phase = "train"
    total_iter["train"] = len(train_dataloader) * epoch
    log[phase][epoch] = {
            # info
            "forward": [],
            "backward": [],
            "fetch": [],
            "iter_time": [],
            # loss (float, not torch.cuda.FloatTensor)
            "loss": [],
            # constraint loss (float, not torch.cuda.FloatTensor)
            "acc": [],
            "miou": []
        }

    ## Prepare everything for training
    train_loss = []
    if opt.use_proxy_loss:
        model2d_classifier.train()
    train_loss_2d = []
    model.train()
    start = time.time()
    model2d_trainable.train()
    num_classes = opt.num_classes # idk why this is necessary, otherwise num_classes is referenced before assignment

    # initialize Tensors for depth, color, camera pose, labels for projection pass
    depth_images = torch.cuda.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
    color_images = torch.cuda.FloatTensor(batch_size * num_images, 3, input_image_dims[1], input_image_dims[0])
    camera_poses = torch.cuda.FloatTensor(batch_size * num_images, 4, 4)
    label_images = torch.cuda.LongTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])

    tempTime = time.time()

    for t, data in enumerate(train_dataloader):
        if(t == 5):
            break
        ## Logs for current training iteration
        running_log = {
            # loss
            "loss": 0,
            # acc
            "acc": 0,
            "miou": 0
        }

        ## Load data
        points, test, targets, frames, weights, fetch_time = data
        points, test, targets, weights = points.cuda(), test.cuda(), targets.cuda(), weights.cuda()
        log[phase][epoch]["fetch"].append(fetch_time)



        start_forward = time.time()
        # targets = torch.autograd.Variable(labels[v].cuda())
        # valid targets

        ## Only keep those Points that are labeled as a class that has at least one appearance
        #TODO: Do we need this masking? There should only be points that are valid
        #TODO: (The same also for test())
        mask = targets.view(-1).data.clone()
        for k in range(num_classes):
            if criterion_weights[k] == 0:
                mask[mask.eq(k)] = 0 # excludes all objects that are not contained in class list
        maskindices = mask.nonzero().squeeze()
        if len(maskindices.shape) == 0:
            continue

        ## Load images, camera poses and labels
        ## frames contains the numbers of the images that correspond to the respective scene chunk
        data_util.load_frames_multi(opt.data_path_2d, frames, depth_images, color_images, camera_poses, color_mean, color_std)

        ## Compute projection mapping
        ## Outputs are the numbers of feature-pixels that correspond to each point, as well as points that correspond to each feature-pixel
        #TODO: Is this comment correct?
        points_projection = torch.repeat_interleave(points, num_images, dim=0)
        proj_mapping = [projection.compute_projection(p, d, c, num_points) for p, d, c in zip(points_projection, depth_images, camera_poses)]
        if None in proj_mapping: # invalid sample
            # print('(invalid sample)')
            continue
        proj_mapping = list(zip(*proj_mapping))
        proj_ind_3d = torch.stack(proj_mapping[0])
        proj_ind_2d = torch.stack(proj_mapping[1])

        #TODO: Same again: Do we need this masking? Here probably yes; I don't know the classes of the images, but probably they are not completely the same as ours
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
        log[phase][epoch]["forward"].append(time.time() - start_forward)
        pred = output
        num_classes = pred.size(2)
        loss = criterion(pred.contiguous().view(-1, num_classes), targets.view(-1).cuda(), weights.view(-1).cuda())
        pred = torch.argmax(pred,2)

        running_log["acc"] = pred.eq(targets).sum().item() / pred.view(-1).size(0)
        start = time.time()

        running_log["loss"] = loss
        # loss = criterion(output.view(-1, num_classes), targets.view(-1))
        # _, loss, _ = model_fn(model, (points, targets), imageft, torch.autograd.Variable(proj_ind_3d), torch.autograd.Variable(proj_ind_2d))

        ##computation of miou
        miou = []
        for i in range(21):
            # if i == 0: continue
            pred_ids = torch.arange(pred.view(-1).size(0))[pred.view(-1) == i].tolist()
            target_ids = torch.arange(targets.view(-1).size(0))[targets.view(-1) == i].tolist()
            if len(target_ids) == 0:
                if (len(pred_ids) != 0):  ## added these 2 lines: Before, we did not incorporate classes that were predicted, but did not appear.
                    miou.append(0)  ## Not sure if it makes sense to include this
                continue
            num_correct = len(set(pred_ids).intersection(set(target_ids)))
            num_union = len(set(pred_ids).union(set(target_ids)))
            miou.append(num_correct / (num_union + 1e-8))

        running_log["miou"] = np.mean(miou)
        ## endo of miou



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
        log[phase][epoch]["backward"].append(time.time() - start)
        log[phase][epoch]["loss"].append(running_log["loss"].item())
        log[phase][epoch]["acc"].append(running_log["acc"])
        log[phase][epoch]["miou"].append(running_log["miou"])


        # confusion
        y = output.data
        y = y.view(int(y.nelement()/y.size(2)), num_classes)[:, :-1]
        _, predictions = y.max(1)
        predictions = predictions.view(-1)
        k = targets.data.view(-1)
        # computes the confustion matrix
        confusion.add(torch.index_select(predictions, 0, maskindices.cuda()), torch.index_select(k, 0, maskindices))
        if(t % 10 == 0):
            #print(t,"/", len(train_dataloader))
            # print("10 iterations took us " + str(time.time() - tempTime))
            tempTime = time.time()
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

        iter_time = log[phase][epoch]["fetch"][-1]
        iter_time += log[phase][epoch]["forward"][-1]
        iter_time += log[phase][epoch]["backward"][-1]
        log[phase][epoch]["iter_time"].append(iter_time)

        if (t + 1) % 10 == 0:
            train_report(epoch)
        global_iter_id += 1

    end = time.time()
    took = end - start
    evaluate_confusion(confusion, train_loss, epoch, iter, took, 'Train', log_file)
    if opt.use_proxy_loss:
        evaluate_confusion(confusion2d, train_loss_2d, epoch, iter, took, 'Train2d', log_file_2d)
    return train_loss, iter, train_loss_2d


def test(epoch, iter, log_file, val_dataloader, log_file_2d):
    total_iter["val"] = len(val_dataloader) * epoch
    phase = "val"
    log[phase][epoch] = {
        # info
        "forward": [],
        "backward": [],
        "fetch": [],
        "iter_time": [],
        # loss (float, not torch.cuda.FloatTensor)
        "loss": [],
        # constraint loss (float, not torch.cuda.FloatTensor)
        "acc": [],
        "miou": []
    }

    test_loss = []
    test_loss_2d = []
    model.eval()
    model2d_fixed.eval()
    model2d_trainable.eval()
    if opt.use_proxy_loss:
        model2d_classifier.eval()
    start = time.time()
    num_classes = opt.num_classes

    #points, labels, frames = data_util.load_hdf5_data(val_file, num_classes)
    #num_points = points.shape[1]

    #frames = frames[:, :2+num_images]
    #num_samples = points.shape[0]
    # shuffle
    #indices = torch.randperm(num_samples).long().split(batch_size)
    # remove last mini-batch so that all the batches have equal size
    #indices = indices[:-1]

    with torch.no_grad():
        depth_images = torch.cuda.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
        color_images = torch.cuda.FloatTensor(batch_size * num_images, 3, input_image_dims[1], input_image_dims[0])
        camera_poses = torch.cuda.FloatTensor(batch_size * num_images, 4, 4)
        label_images = torch.cuda.LongTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])


        for t,data in enumerate(val_dataloader):
            running_log = {
                # loss
                "loss": 0,
                # acc
                "acc": 0,
                "miou": 0
            }
            points, test, targets, frames, weights, fetch_time = data
            points, test, targets, weights = points.cuda(), test.cuda(), targets.cuda(), weights.cuda()
            log[phase][epoch]["fetch"].append(fetch_time)

            num_points = points.shape[1]

            frames = frames[:, :2 + num_images]
            #num_samples = points.shape[0]
            #targets = labels[v].cuda()
            # valid targets
            mask = targets.view(-1).data.clone()
            for k in range(num_classes):
                if criterion_weights[k] == 0:
                    mask[mask.eq(k)] = 0
            maskindices = mask.nonzero().squeeze()
            if len(maskindices.shape) == 0:
                continue
            # get 2d data
            data_util.load_frames_multi(opt.data_path_2d, frames, depth_images, color_images, camera_poses, color_mean, color_std)
            if opt.use_proxy_loss:
                data_util.load_label_frames(opt.data_path_2d, frames, label_images, num_classes)
                mask2d = label_images.view(-1).clone()
                for k in range(num_classes):
                    if criterion_weights[k] == 0:
                        mask2d[mask2d.eq(k)] = 0
                mask2d = mask2d.nonzero().squeeze()
                if (len(mask2d.shape) == 0):
                    continue  # nothing to optimize for here

            # compute projection mapping
            proj_mapping = [projection.compute_projection(p, d, c, num_points) for p, d, c in zip(points, depth_images, camera_poses)]
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
            input3d = points.cuda()
            output = model(input3d, imageft, proj_ind_3d, proj_ind_2d)
            preds = torch.argmax(output, 2)
            running_log["acc"] = preds.eq(targets).sum().item() / preds.view(-1).size(0)

            loss = criterion(output.view(-1, num_classes), targets.view(-1), weights.view(-1))
            running_log["loss"] = loss

            ##Computation of Miou
            miou = []
            for i in range(21):
                # if i == 0: continue
                pred_ids = torch.arange(preds.view(-1).size(0))[preds.view(-1) == i].tolist()
                target_ids = torch.arange(targets.view(-1).size(0))[targets.view(-1) == i].tolist()
                if len(target_ids) == 0:
                    if (len(pred_ids) != 0):  ## added these 2 lines: Before, we did not incorporate classes that were predicted, but did not appear.
                        miou.append(0)  ## Not sure if it makes sense to include this
                    continue
                num_correct = len(set(pred_ids).intersection(set(target_ids)))
                num_union = len(set(pred_ids).union(set(target_ids)))
                miou.append(num_correct / (num_union + 1e-8))

            running_log["miou"] = np.mean(miou)
            ##End of Miou

            log[phase][epoch]["loss"].append(running_log["loss"].item())
            log[phase][epoch]["acc"].append(running_log["acc"])
            log[phase][epoch]["miou"].append(running_log["miou"])

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
            y = y.view(int(y.nelement()/y.size(2)), num_classes)[:, :-1]
            _, predictions = y.max(1)
            predictions = predictions.view(-1)
            k = targets.data.view(-1)
            confusion_val.add(torch.index_select(predictions, 0, maskindices), torch.index_select(k, 0, maskindices))
            if(epoch % 5 == 0):
                model_root = os.path.join(CONF.OUTPUT_ROOT, stamp)
                torch.save(model, os.path.join(model_root, str(epoch) + "model.pth"))

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
        num = conf[c, :].sum() # number of points with ground truth c (TP + FN)
        valids[c] = -1 if num == 0 else float(conf[c][c]) / float(num) # TP / (TP + FN)
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
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    tb_path = os.path.join(CONF.OUTPUT_ROOT, stamp, "tensorboard")

    global log_writer
    log_writer = SummaryWriter(tb_path)

    global global_iter_id
    global_iter_id = 0

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
        #for k in range(len(train_dataloader)):
        k = 1234
        print('Epoch: {}\tFile: {}/{}\t{}'.format(epoch, k, len(train_dataloader), train_dataloader))
        loss, iter, loss2d = train(epoch, iter, log_file, train_dataloader, log_file_2d)
        train_loss.extend(loss)
        if loss2d:
             train2d_loss.extend(loss2d)
        if has_val: #and k % num_files_per_val == 0:
            # val_index = torch.randperm(len(val_dataloader))[0]
            loss, loss2d = test(epoch, iter, log_file_val, val_dataloader, log_file_2d_val)
            val_loss.extend(loss)
            if loss2d:
                 val2d_loss.extend(loss2d)

        epoch_report(epoch)
        dump_log(epoch)
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

def train_report(epoch_id):
    fetch_time = [time for time in log["train"][epoch_id]["fetch"]]
    forward_time = [time for time in log["train"][epoch_id]["forward"]]
    backward_time = [time for time in log["train"][epoch_id]["backward"]]
    iter_time = [time for time in log["train"][epoch_id]["iter_time"]]
    mean_train_time = np.mean(iter_time[0].numpy())
    mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time[0], forward_time)])
    #eta_sec = (total_iter["train"] - global_iter_id - 1) * mean_train_time
    #eta_sec += len(self.dataloader["val"]) * (self.epoch - epoch_id) * mean_est_val_time
    #eta = decode_eta(eta_sec)

    # print report
    iter_report = iter_report_template.format(
        global_iter_id=global_iter_id + 1,
        total_iter=total_iter["train"],
        train_loss=round(np.mean([loss for loss in log["train"][epoch_id]["loss"]]), 5),
        train_acc=round(np.mean([loss for loss in log["train"][epoch_id]["acc"]]), 5),
        train_miou=round(np.mean([loss for loss in log["train"][epoch_id]["miou"]]), 5),
        mean_fetch_time=round(np.mean(fetch_time[0].numpy()), 5),
        mean_forward_time=round(np.mean(forward_time), 5),
        mean_backward_time=round(np.mean(backward_time), 5),
        mean_iter_time=round(np.mean(iter_time[0].numpy()), 5),
        #eta_h=eta["h"],
        #eta_m=eta["m"],
        #eta_s=eta["s"]
    )

def dump_log(epoch_id):
    # loss
    print("Writing everything.")
    log_writer.add_scalars(
        "log/{}".format("loss"),
        {
            "train": np.mean([loss for loss in log["train"][epoch_id]["loss"]]),
            "val": np.mean([loss for loss in log["val"][epoch_id]["loss"]])
        },
        epoch_id
    )

    # eval
    log_writer.add_scalars(
        "eval/{}".format("acc"),
        {
            "train": np.mean([acc for acc in log["train"][epoch_id]["acc"]]),
            "val": np.mean([acc for acc in log["val"][epoch_id]["acc"]])
        },
        epoch_id
    )
    log_writer.add_scalars(
        "eval/{}".format("miou"),
        {
            "train": np.mean([miou for miou in log["train"][epoch_id]["miou"]]),
            "val": np.mean([miou for miou in log["val"][epoch_id]["miou"]])
        },
        epoch_id
    )


def epoch_report(epoch_id):
    print("epoch [{}/{}] done...".format(epoch_id+1, 20))
    epoch_report = epoch_report_template.format(
        train_loss=round(np.mean([loss for loss in log["train"][epoch_id]["loss"]]), 5),
        train_acc=round(np.mean([acc for acc in log["train"][epoch_id]["acc"]]), 5),
        train_miou=round(np.mean([miou for miou in log["train"][epoch_id]["miou"]]), 5),
        val_loss=round(np.mean([loss for loss in log["val"][epoch_id]["loss"]]), 5),
        val_acc=round(np.mean([acc for acc in log["val"][epoch_id]["acc"]]), 5),
        val_miou=round(np.mean([miou for miou in log["val"][epoch_id]["miou"]]), 5),
    )
    print(epoch_report)

if __name__ == '__main__':
    main()


