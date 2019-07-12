
import argparse
import torch
from scipy import misc
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

from enet import create_enet_for_3d


ENET_TYPES = {'scannet': (41, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}  #classes, color mean/std

# params
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', default=42, help='#classes')
parser.add_argument('--model2d_type', default='scannet', help='which enet (scannet)')
parser.add_argument('--model2d_path', required=False, default='scannetv2_enet.pth', help='path to enet model')
parser.add_argument('--data_path_2d', required=False, default='/media/lorenzlamm/My Book/Scannet/out_images', help='path to 2d train data')


parser.set_defaults(use_proxy_loss=False)
opt = parser.parse_args()
assert opt.model2d_type in ENET_TYPES


# create enet models

num_classes = opt.num_classes
model2d_fixed, model2d_trainable, model2d_classifier = create_enet_for_3d(ENET_TYPES[opt.model2d_type], opt.model2d_path, num_classes)
model2d_fixed = model2d_fixed.cuda()
model2d_trainable = model2d_trainable.cuda()
model2d_classifier = model2d_classifier.cuda()
def resize_crop_image(image, new_image_dims):
    image_dims = [image.shape[1], image.shape[0]]
    if image_dims == new_image_dims:
        return image
    resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
    image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
    image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
    image = np.array(image)
    return image

input_image_dims = [328, 256]
color_mean = ENET_TYPES[opt.model2d_type][1]
color_std = ENET_TYPES[opt.model2d_type][2]
color_images = torch.cuda.FloatTensor(1 * 1, 3, input_image_dims[1], input_image_dims[0])

color_image = misc.imread("/home/lorenzlamm/Dokumente/final_network/adl4cv/840.jpg")
color_image = resize_crop_image(color_image, input_image_dims)
color_image = np.transpose(color_image, [2, 0, 1])  # move feature to front
normalize = transforms.Normalize(mean=color_mean, std=color_std)
color_image = normalize(torch.Tensor(color_image.astype(np.float32) / 255.0))

imageft_fixed = model2d_fixed(torch.autograd.Variable(color_images))
print(imageft_fixed.shape)
plt.figure()
plt.imshow(imageft_fixed[0][100].cpu())
plt.show()
imageft = model2d_trainable(imageft_fixed)
plt.figure()
plt.imshow(imageft[0][100].detach().cpu())
plt.show()
