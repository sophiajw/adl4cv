import os
from easydict import EasyDict

CONF = EasyDict()

# BASE PATH
<<<<<<< HEAD
CONF.ROOT = "/home/lorenzlamm/Dokumente/DavesPointnetClone/Pointnet2.ScanNet" # TODO change this
CONF.SCANNET_DIR =  "/home/lorenzlamm/Downloads/scans/scans" # TODO change this
CONF.SCENE_NAMES = os.listdir('/home/lorenzlamm/Downloads/scans/scans') # TODO change
=======
CONF.ROOT = "/workspace/pointnet_train/Pointnet2.ScanNet" # TODO change this
CONF.SCANNET_DIR =  "/mnt/canis/Datasets/ScanNet/public/v2/scans" # TODO change this
#CONF.SCENE_NAMES = os.listdir('/mnt/canis/Datasets/ScanNet/public/v1/scans') # TODO change
>>>>>>> 492875351ce329ba901829f996103c7bd052e9be

CONF.PREP = os.path.join(CONF.ROOT, "preprocessing")
CONF.PREP_SCANS = os.path.join(CONF.PREP, "scannet_scenes")
CONF.OUTPUT_ROOT = os.path.join(CONF.ROOT, "outputs")

CONF.SCANNETV2_TRAIN = os.path.join(CONF.ROOT, "data/scannetv2_train.txt")
CONF.SCANNETV2_VAL = os.path.join(CONF.ROOT, "data/scannetv2_val.txt")
CONF.SCANNETV2_FILE = os.path.join(CONF.PREP_SCANS, "{}.npy") # scene_id
