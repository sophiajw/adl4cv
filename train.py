"""
Based on https://github.com/angeladai/3DMV and https://github.com/daveredrum/Pointnet2.ScanNet
"""
import argparse
import os, sys, time
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

from utils import util
from data import data_util
from model import BeachNet
from enet import create_enet_for_3d
from utils.projection import ProjectionHelper
from data.load_data import DataLoader
 
from tensorboardX import SummaryWriter
sys.path.append(".")

log = {phase: {} for phase in ["train", "val"]}
ENET_TYPES = {'scannet': (41, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}  #classes, color mean/std 
global_iter_id = 0
total_iter = {}

visualize_test_scene = False # For computation of test outputs for visualization
eval_flag = True # For evaluation on test Scene
visual_flag = False # activate for visualizing a scene



# Templates for progress reports while training (and after training has finished)

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

EPOCH_REPORT_TEMPLATE_PROXY = """
------------------------summary------------------------
[train] train_loss: {train_loss}
[train] train_acc: {train_acc}
[train] train_miou: {train_miou}
[train] train_2D_loss: {train_2D_loss}
[val]   val_loss: {val_loss}
[val]   val_acc: {val_acc}
[val]   val_miou: {val_miou}
[val]   val_2D_loss: {val_2D_loss}
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
epoch_report_template_proxy = EPOCH_REPORT_TEMPLATE
best_report_template = BEST_REPORT_TEMPLATE



# Parser arguments

parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--input_folder_3d', required=False,
                    default='/workspace/beachnet_train/bn_train_data')
parser.add_argument('--root_directory', default='/workspace/beachnet_train/adl4cv', help='directory of train.py')
parser.add_argument('--val_data_list', default='', help='path to file list of h5 val data')
parser.add_argument('--output', default='./logs_evaluation', help='folder to output model checkpoints')
parser.add_argument('--data_path_2d', required=False, default='/workspace/beachnet_train/bn_train_data',
                        help='path to 2d train data')

# train params
parser.add_argument('--num_classes', default=21, help='#classes')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--max_epoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--lr_pointnet', type=float, default=1e-3, help='Initial learning rate for PointNet [default: 1e-2]')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum, default=0.9')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay, default=0.0005')
parser.add_argument('--weight_decay_pointnet', type=float, default=0, help='L2 regularization coeff [default: 0.0]')
parser.add_argument('--model2d_type', default='scannet', help='which enet (scannet)')
parser.add_argument('--model2d_path', required=False,
                        default='/workspace/beachnet_train/bn_train_data/scannetv2_enet.pth', help='path to enet model')
parser.add_argument('--use_proxy_loss', dest='use_proxy_loss', action='store_true')
parser.add_argument('--num_points', default=4096, help='number of points in one sample')

# 2d/3d
parser.add_argument('--accuracy', type=float, default=0.05, help='accuracy of point projection (in meters)')
parser.add_argument('--depth_min', type=float, default=0.4, help='min depth (in meters)')
parser.add_argument('--depth_max', type=float, default=4.0, help='max depth (in meters)')
# scannet intrinsic params
parser.add_argument('--intrinsic_image_width', type=int, default=640, help='2d image width')
parser.add_argument('--intrinsic_image_height', type=int, default=480, help='2d image height')
parser.add_argument('--fx', type=float, default=577.870605, help='intrinsics')
parser.add_argument('--fy', type=float, default=577.870605, help='intrinsics')
parser.add_argument('--mx', type=float, default=319.5, help='intrinsics')
parser.add_argument('--my', type=float, default=239.5, help='intrinsics')


parser.set_defaults(use_proxy_loss=True)
opt = parser.parse_args()
assert opt.model2d_type in ENET_TYPES

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(opt.gpu)

# create camera intrinsics
input_image_dims = [328, 256]
proj_image_dims = [41, 32]
intrinsic = util.make_intrinsic(opt.fx, opt.fy, opt.mx, opt.my)
intrinsic = util.adjust_intrinsic(intrinsic, [opt.intrinsic_image_width, opt.intrinsic_image_height], proj_image_dims)
intrinsic = intrinsic.cuda()
batch_size = opt.batch_size
num_images = 3
color_mean = ENET_TYPES[opt.model2d_type][1]
color_std = ENET_TYPES[opt.model2d_type][2]
input_channels = 128
num_points = opt.num_points

# Start initialization
print("initializing...")
stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
root = os.path.join(opt.root_directory, stamp)
os.makedirs(root, exist_ok=True)


# create enet and pointnet++ models
num_classes = opt.num_classes
model2d_fixed, model2d_trainable, model2d_classifier = create_enet_for_3d(ENET_TYPES[opt.model2d_type], opt.model2d_path, num_classes)
model = BeachNet(num_classes, num_images, input_channels, intrinsic, proj_image_dims, opt.depth_min, opt.depth_max,
                 opt.accuracy, fusion=True, fuseAtPosition = 2, fuse_no_ft_pn=False, pointnet_pointnet=False)
projection = ProjectionHelper(intrinsic, opt.depth_min, opt.depth_max, proj_image_dims, opt.accuracy)

# create loss
criterion = util.WeightedCrossEntropyLoss()
criterion2d = torch.nn.CrossEntropyLoss().cuda()

# move to gpu
model2d_fixed = model2d_fixed.cuda()
model2d_trainable = model2d_trainable.cuda()
model2d_classifier = model2d_classifier.cuda()
model = model.cuda()
criterion = criterion.cuda()

# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr_pointnet, weight_decay=opt.weight_decay_pointnet)
optimizer2d = torch.optim.SGD(model2d_trainable.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
if opt.use_proxy_loss:
    optimizer2dc = torch.optim.SGD(model2d_classifier.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)


if(not eval_flag and not visual_flag):
    train_dataset = DataLoader(num_points, root=opt.input_folder_3d, train=True)
    val_dataset = DataLoader(num_points, root=opt.input_folder_3d, train=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        pin_memory=True,
        num_workers=8,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }
    train_examples = len(train_dataset)
    val_examples = len(val_dataset)


    _SPLITTER = ','


    print("\n[info]")
    print("Train examples: {}".format(train_examples))
    print("Evaluation examples: {}".format(val_examples))
    print("Start training...\n")

def train(epoch, train_dataloader):
    """
    Training of Epoch
    :param epoch: int
    :param train_dataloader: training dataloader
    """
    global global_iter_id
    batch_size = train_dataloader.batch_size

    # Parameters for logging
    phase = "train"
    total_iter["train"] = len(train_dataloader) * epoch
    log[phase][epoch] = {
            "forward": [],
            "backward": [],
            "fetch": [],
            "iter_time": [],
            "loss": [],
            "acc": [],
            "miou": [],
            "loss_2d": []
        }

    # Prepare everything for training
    train_loss = []
    train_loss_2d = []
    model2d_trainable.train()
    model.train()
    if opt.use_proxy_loss:
        model2d_classifier.train()
    num_classes = opt.num_classes

    # initialize Tensors for depth, color, camera pose, labels for projection pass
    depth_images = torch.cuda.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
    color_images = torch.cuda.FloatTensor(batch_size * num_images, 3, input_image_dims[1], input_image_dims[0])
    camera_poses = torch.cuda.FloatTensor(batch_size * num_images, 4, 4)
    label_images = torch.cuda.LongTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])

    for t, data in enumerate(train_dataloader):
        # Logs for current training iteration
        running_log = {
            "loss": 0,
            "acc": 0,
            "miou": 0
        }

        # Load data
        points, test, targets, frames, weights, fetch_time = data
        points, test, targets, weights = points.cuda(), test.cuda(), targets.cuda(), weights.cuda()
        log[phase][epoch]["fetch"].append(fetch_time)
        start_forward = time.time()

        # Load images, camera poses and labels
        # frames contains the numbers of the images that correspond to the respective scene chunk
        data_util.load_frames_multi(opt.data_path_2d, frames, depth_images, color_images, camera_poses, color_mean, color_std)

        # If we use proxy loss, we also load the label images for our input images
        if opt.use_proxy_loss:
            data_util.load_label_frames(opt.data_path_2d, frames, label_images, num_classes)
            mask2d = label_images.view(-1).clone()
            mask2d = mask2d.nonzero().squeeze()
            if(mask2d.shape == torch.Size([0])):
                continue

        # Compute projection mapping
        # Outputs are the numbers of feature-pixels that correspond to each point, as well as points that correspond to each feature-pixel
        points_projection = torch.repeat_interleave(points, num_images, dim=0) # For each scene chunk, we have num_images images. We repeat each point cloud num_images times to compute the projection
        proj_mapping = [projection.compute_projection(p, d, c, num_points) for p, d, c in
                        zip(points_projection, depth_images, camera_poses)]
        if None in proj_mapping:  # invalid sample
            continue
        proj_mapping = list(zip(*proj_mapping))
        proj_ind_3d = torch.stack(proj_mapping[0])
        proj_ind_2d = torch.stack(proj_mapping[1])

        # 2d forward pass
        imageft_fixed = model2d_fixed(torch.autograd.Variable(color_images))
        imageft = model2d_trainable(imageft_fixed)
        if opt.use_proxy_loss:
            ft2d = model2d_classifier(imageft)
            ft2d = ft2d.permute(0, 2, 3, 1).contiguous()
        # 3d forward pass
        input3d = torch.autograd.Variable(points.cuda())
        output = model(input3d, imageft, torch.autograd.Variable(proj_ind_3d), torch.autograd.Variable(proj_ind_2d))
        log[phase][epoch]["forward"].append(time.time() - start_forward)
        preds = torch.argmax(output, 2)
        loss = criterion(output.contiguous().view(-1, num_classes), targets.view(-1).cuda(), weights.view(-1).cuda())
        train_loss.append(loss.item())
        running_log["loss"] = loss
        running_log["acc"] = preds.eq(targets).sum().item() / preds.view(-1).size(0)
        start = time.time()

        # Computation of mIoU
        miou = []
        for i in range(21):
            # if i == 0: continue
            pred_ids = torch.arange(preds.view(-1).size(0))[preds.view(-1) == i].tolist()
            target_ids = torch.arange(targets.view(-1).size(0))[targets.view(-1) == i].tolist()
            if len(target_ids) == 0:
                if (len(pred_ids) != 0): # If there are no targets of this class, but predictions for this class, append 0 to mIoUs
                    miou.append(0)
                continue
            num_correct = len(set(pred_ids).intersection(set(target_ids)))
            num_union = len(set(pred_ids).union(set(target_ids)))
            miou.append(num_correct / (num_union + 1e-8))

        running_log["miou"] = np.mean(miou)
        # end of miou

        # 3D Backward step
        optimizer.zero_grad()
        optimizer2d.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer2d.step()

        # If we use proxy loss, do 2D Backward step
        if opt.use_proxy_loss:
            loss2d = criterion2d(ft2d.view(-1, num_classes), torch.autograd.Variable(label_images.view(-1)))
            train_loss_2d.append(loss2d.item())
            optimizer2d.zero_grad()
            optimizer2dc.zero_grad()
            loss2d.backward()
            optimizer2dc.step()
            optimizer2d.step()

        # Log everything
        log[phase][epoch]["backward"].append(time.time() - start)
        log[phase][epoch]["loss"].append(running_log["loss"].item())
        log[phase][epoch]["acc"].append(running_log["acc"])
        log[phase][epoch]["miou"].append(running_log["miou"])
        if(opt.use_proxy_loss):
            log[phase][epoch]["loss_2d"].append(loss2d.item())
        iter_time = log[phase][epoch]["fetch"][-1]
        iter_time += log[phase][epoch]["forward"][-1]
        iter_time += log[phase][epoch]["backward"][-1]
        log[phase][epoch]["iter_time"].append(iter_time)
        global_iter_id += 1

    if (t + 1) % 50 == 0:
        train_report(epoch)



def test(epoch, val_dataloader):
    """
    Validation of the epoch
    :param epoch: int
    :param val_dataloader: validation dataloader
    """
    batch_size = val_dataloader.batch_size
    total_iter["val"] = len(val_dataloader) * epoch
    phase = "val"
    log[phase][epoch] = {
        "forward": [],
        "backward": [],
        "fetch": [],
        "iter_time": [],
        "loss": [],
        "acc": [],
        "miou": [],
        "loss_2d": []
    }
    if(eval_flag):
        model2d_trainable_dict = torch.load("/media/lorenzlamm/My Book/beachnet_training_results/logs_fuse/logs_fuse/model2d-epoch-23.pth")
        model2d_fixed_dict = torch.load("/media/lorenzlamm/My Book/beachnet_training_results/logs_fuse/logs_fuse/model2dfixed.pth")
        model_dict = torch.load("/media/lorenzlamm/My Book/beachnet_training_results/logs_fuse/logs_fuse/model-epoch-23.pth")

        model.load_state_dict(model_dict)
        model2d_fixed.load_state_dict(model2d_fixed_dict)
        model2d_trainable.load_state_dict(model2d_trainable_dict)

    test_loss = []
    test_loss_2d = []
    model.eval()
    model2d_fixed.eval()
    model2d_trainable.eval()
    if opt.use_proxy_loss:
        model2d_classifier.eval()
    num_classes = opt.num_classes

    with torch.no_grad():
        depth_images = torch.cuda.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
        color_images = torch.cuda.FloatTensor(batch_size * num_images, 3, input_image_dims[1], input_image_dims[0])
        camera_poses = torch.cuda.FloatTensor(batch_size * num_images, 4, 4)
        label_images = torch.cuda.LongTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])

        for t, data in enumerate(val_dataloader):
            running_log = {
                "loss": 0,
                "acc": 0,
                "miou": 0
            }
            if(eval_flag):
                print(t, "/", len(val_dataloader))

            # Load data
            points, test, targets, frames, weights, fetch_time = data
            points, test, targets, weights = points.cuda(), test.cuda(), targets.cuda(), weights.cuda()
            log[phase][epoch]["fetch"].append(fetch_time)

            # Load images, camera poses and labels
            # frames contains the numbers of the images that correspond to the respective scene chunk
            data_util.load_frames_multi(opt.data_path_2d, frames, depth_images, color_images, camera_poses, color_mean,
                                        color_std)
            # If we use proxy loss, we also load the label images for our input images
            if opt.use_proxy_loss:
                data_util.load_label_frames(opt.data_path_2d, frames, label_images, num_classes)
                mask2d = label_images.view(-1).clone()
                mask2d = mask2d.nonzero().squeeze()
                if (mask2d.shape == torch.Size([0])):
                    continue

            # Compute projection mapping
            # Outputs are the numbers of feature-pixels that correspond to each point, as well as points that correspond to each feature-pixel
            points_projection = torch.repeat_interleave(points, num_images, dim=0)
            proj_mapping = [projection.compute_projection(p, d, c, num_points) for p, d, c in
                            zip(points_projection, depth_images, camera_poses)]
            if None in proj_mapping:  # invalid sample
                continue
            proj_mapping = list(zip(*proj_mapping))
            proj_ind_3d = torch.stack(proj_mapping[0])
            proj_ind_2d = torch.stack(proj_mapping[1])

            # 2d forward pass
            imageft_fixed = model2d_fixed(color_images)
            imageft = model2d_trainable(imageft_fixed)
            if opt.use_proxy_loss:
                ft2d = model2d_classifier(imageft)
                ft2d = ft2d.permute(0, 2, 3, 1).contiguous()
            # 3d forward pass
            input3d = points.cuda()
            output = model(input3d, imageft, proj_ind_3d, proj_ind_2d)
            preds = torch.argmax(output, 2)
            loss = criterion(output.view(-1, num_classes), targets.view(-1), weights.view(-1))
            test_loss.append(loss.item())
            running_log["acc"] = preds.eq(targets).sum().item() / preds.view(-1).size(0)
            running_log["loss"] = loss

            #Computation of mIoU
            miou = []
            for i in range(21):
                # if i == 0: continue
                pred_ids = torch.arange(preds.view(-1).size(0))[preds.view(-1) == i].tolist()
                target_ids = torch.arange(targets.view(-1).size(0))[targets.view(-1) == i].tolist()
                if len(target_ids) == 0:
                    if (len(pred_ids) != 0):
                        miou.append(0)
                    continue
                num_correct = len(set(pred_ids).intersection(set(target_ids)))
                num_union = len(set(pred_ids).union(set(target_ids)))
                miou.append(num_correct / (num_union + 1e-8))
            running_log["miou"] = np.mean(miou)
            #End of mIoU

            # If we use proxy loss, also compute 2D loss
            if opt.use_proxy_loss:
                loss2d = criterion2d(ft2d.view(-1, num_classes), label_images.view(-1))
                test_loss_2d.append(loss2d.item())

            log[phase][epoch]["loss"].append(running_log["loss"].item())
            log[phase][epoch]["acc"].append(running_log["acc"])
            log[phase][epoch]["miou"].append(running_log["miou"])
            if(opt.use_proxy_loss):
                log[phase][epoch]["loss_2d"].append(loss2d.item())



def test_for_visual(vis_dataloader):
    r"""
    Computes the predicted point cloud for the input point cloud (used for visualization)
    :param vis_dataloader: Dataloader (should contain only scene chunk from 1 single scene)
    :return: out_scene: point cloud with predicted labels, [Torch Tensor Size: [npoints, 4]]
             gt_scene: point cloud with ground truth labels, [Torch Tensor Size: [npoints, 4]]
    """

    # Load pretrained models
    model2d_trainable_dict = torch.load("/home/lorenzlamm/Dokumente/final_new/adl4cv/models/model2d-epoch-15.pth")
    model2d_fixed_dict = torch.load("/home/lorenzlamm/Dokumente/final_new/adl4cv/models/model2dfixed.pth")
    model_dict = torch.load("/home/lorenzlamm/Dokumente/final_new/adl4cv/models/model-epoch-15.pth")
    model.load_state_dict(model_dict)
    model2d_fixed.load_state_dict(model2d_fixed_dict)
    model2d_trainable.load_state_dict(model2d_trainable_dict)
    model.eval()
    model2d_fixed.eval()
    model2d_trainable.eval()
    if opt.use_proxy_loss:
        model2d_classifier.eval()
    num_classes = opt.num_classes

    num_points = opt.num_points
    scene = torch.zeros(len(vis_dataloader)*num_points)
    scene_labels = torch.zeros(len(vis_dataloader)*num_points)
    scene_points = torch.zeros(len(vis_dataloader)*num_points,3)
    with torch.no_grad():
        # initialize Tensors for depth, color, camera pose, labels for projection pass
        depth_images = torch.cuda.FloatTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])
        color_images = torch.cuda.FloatTensor(batch_size * num_images, 3, input_image_dims[1], input_image_dims[0])
        camera_poses = torch.cuda.FloatTensor(batch_size * num_images, 4, 4)
        label_images = torch.cuda.LongTensor(batch_size * num_images, proj_image_dims[1], proj_image_dims[0])

        for t, data in enumerate(vis_dataloader):
            # Load data
            points, test, targets, frames, weights, fetch_time = data
            scene_points[t*num_points:(t+1)*num_points] = points
            points, test, targets, weights = points.cuda(), test.cuda(), targets.cuda(), weights.cuda()
            scene_labels[t*num_points:(t+1)*num_points] = targets.view(-1)

            # get 2d data
            data_util.load_frames_multi(opt.data_path_2d, frames, depth_images, color_images, camera_poses, color_mean,
                                        color_std)
            # If we use proxy loss, we also load the label images for our input images
            if opt.use_proxy_loss:
                data_util.load_label_frames(opt.data_path_2d, frames, label_images, num_classes)
                mask2d = label_images.view(-1).clone()
                mask2d = mask2d.nonzero().squeeze()
                if (len(mask2d.shape) == 0):
                    continue

            # Compute projection mapping
            # Outputs are the numbers of feature-pixels that correspond to each point, as well as points that correspond to each feature-pixel
            proj_mapping = [projection.compute_projection(p, d, c, num_points) for p, d, c in
                            zip(points, depth_images, camera_poses)]
            if None in proj_mapping:  # invalid sample
                continue
            proj_mapping = list(zip(*proj_mapping))
            proj_ind_3d = torch.stack(proj_mapping[0])
            proj_ind_2d = torch.stack(proj_mapping[1])

            # 2d forward pass
            imageft_fixed = model2d_fixed(color_images)
            imageft = model2d_trainable(imageft_fixed)
            if opt.use_proxy_loss:
                ft2d = model2d_classifier(imageft)
                ft2d = ft2d.permute(0, 2, 3, 1).contiguous()
            # 3D forward pass
            input3d = points.cuda()
            output = model(input3d, imageft, proj_ind_3d, proj_ind_2d)
            preds = torch.argmax(output, 2)
            # The predicted labels for the current scene chunk
            scene[t*4096:(t+1)*4096] = preds.squeeze()

    scene = scene.unsqueeze(1)
    gt_scene = torch.cat((scene_points, scene_labels.unsqueeze(1)),dim=1)
    out_scene = torch.cat((scene_points,scene), dim=1)
    return out_scene, gt_scene


def main():
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    tb_path = os.path.join(opt.root_directory, stamp, "tensorboard")
    global log_writer
    log_writer = SummaryWriter(tb_path)
    global global_iter_id
    global_iter_id = 0
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
    has_val = len(val_dataset) > 0

    # start training
    print('starting training...')
    iter = 0
    for epoch in range(opt.max_epoch):
        train_loss = []
        train2d_loss = []
        val_loss = []
        val2d_loss = []

        print('Epoch: {}\t{}'.format(epoch, train_dataloader))
        # Visualize a test scene
        if(visual_flag):
            scene_nrs = ["0086_00", "0187_01", "0552_01", "0568_00", "0699_00", "0700_01"]
            print("Visualizing...")
            for scene_nr in scene_nrs:
                vis_dataset = DataLoader(num_points, root=opt.input_folder_3d, train=False, test=False, visualize=True, vis_scene = scene_nr)
                vis_dataloader = DataLoader(
                    vis_dataset,
                    batch_size=1,
                    pin_memory=True,
                    num_workers=8,
                    shuffle=False
                )
                pred_scene, gt_scene = test_for_visual(vis_dataloader)
                np.savetxt(os.path.join(opt.output, "scene" + scene_nr + "_pn.txt"), pred_scene, delimiter=',')
                np.savetxt(os.path.join(opt.output, "scene" + scene_nr + "_GT.txt"), gt_scene, delimiter=',')
                print("Saved to disk")
            return
        # if we want to evaluate our model, we feed in the test data
        if(eval_flag):
            test_dataset = DataLoader(num_points, root=opt.input_folder_3d, train=False, test=True)
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=4,
                pin_memory=True,
                num_workers=8,
                shuffle=False
            )
            test(epoch, test_dataloader)
            print("test_loss", np.mean([loss for loss in log["val"][epoch]["loss"]]))
            print("test_acc", np.mean([acc for acc in log["val"][epoch]["acc"]]))
            print("test_miou", np.mean([miou for miou in log["val"][epoch]["miou"]]))
            return
        train(epoch, train_dataloader)
        if has_val:
            test(epoch, val_dataloader)
        epoch_report(epoch)
        dump_log(epoch)

        torch.save(model.state_dict(), os.path.join(opt.output, 'model-epoch-%s.pth' % epoch))
        torch.save(model2d_trainable.state_dict(), os.path.join(opt.output, 'model2d-epoch-%s.pth' % epoch))
        if opt.use_proxy_loss:
            torch.save(model2d_classifier.state_dict(), os.path.join(opt.output, 'model2dc-epoch-%s.pth' % epoch))



def dump_log(epoch_id):
    """
    Writing everything to tensorboard logwriter
    """
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
    """
    Prints the metrics of the previous epoch
    """
    print("epoch [{}/{}] done...".format(epoch_id + 1, 20))
    if(opt.use_proxy_loss):
        epoch_report = epoch_report_template_proxy.format(
            train_loss=round(np.mean([loss for loss in log["train"][epoch_id]["loss"]]), 5),
            train_acc=round(np.mean([acc for acc in log["train"][epoch_id]["acc"]]), 5),
            train_miou=round(np.mean([miou for miou in log["train"][epoch_id]["miou"]]), 5),
            train_2D_loss=round(np.mean([loss for loss in log["train"][epoch_id]["loss_2d"]]), 5),
            val_loss=round(np.mean([loss for loss in log["val"][epoch_id]["loss"]]), 5),
            val_acc=round(np.mean([acc for acc in log["val"][epoch_id]["acc"]]), 5),
            val_miou=round(np.mean([miou for miou in log["val"][epoch_id]["miou"]]), 5),
            val_2D_loss=round(np.mean([loss for loss in log["val"][epoch_id]["loss_2d"]]), 5),
        )
        print(epoch_report)
        return
    epoch_report = epoch_report_template.format(
        train_loss=round(np.mean([loss for loss in log["train"][epoch_id]["loss"]]), 5),
        train_acc=round(np.mean([acc for acc in log["train"][epoch_id]["acc"]]), 5),
        train_miou=round(np.mean([miou for miou in log["train"][epoch_id]["miou"]]), 5),
        val_loss=round(np.mean([loss for loss in log["val"][epoch_id]["loss"]]), 5),
        val_acc=round(np.mean([acc for acc in log["val"][epoch_id]["acc"]]), 5),
        val_miou=round(np.mean([miou for miou in log["val"][epoch_id]["miou"]]), 5),
    )
    print(epoch_report)

def train_report(epoch_id):
    """
    Prints the metrics of the current progress of the optimization
    """
    fetch_time = [time for time in log["train"][epoch_id]["fetch"]]
    forward_time = [time for time in log["train"][epoch_id]["forward"]]
    backward_time = [time for time in log["train"][epoch_id]["backward"]]
    iter_time = [time for time in log["train"][epoch_id]["iter_time"]]

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
    )
    print(iter_report)

if __name__ == '__main__':
    main()


