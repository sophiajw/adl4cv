# adlcv
Advanced Deep Learning for Computer Vision

Our project:

## Multi View Semantic Segmentation on Point Clouds

### Installation
Load the following docker container and additional packages with

    docker pull nvcr.io/nvidia/pytorch:19.06-py3
    nvidia-docker run -it --rm -v /home/ubuntu:/workspace nvcr.io/nvidia/pytorch:19.06-py3
    pip install tensorboardX
    
Install our code with

    python setup.py install
    
### Run our code
The default runs our best model with a fusion after two set abstraction layers of PointNet++
Adapt data, 2d images paths or include them as argument and run the following command to reproduce our results

    python train.py --lr 1e-3 --lr_pointnet 1e-3 --batch_size 8

To run the other model architectures that we implemented change the parameters in the initialization of our model (line 160 in `train.py`)

* Direct concatentation: `fusion=False`, `fuse_no_ft_pn=False`, `pointnet_pointnet=False`
* Process only geometry with PointNet++: `fusion=False`, `fuse_no_ft_pn=True`, `pointnet_pointnet=False`
* PointNet++ in all steps: `fusion=False`, `fuse_no_ft_pn=False`, `pointnet_pointnet=True`
* Fuse after set abstraction layers: `fusion=True`, `fuse_at_position=4`, `fuse_no_ft_pn=False`, `pointnet_pointnet=False`
* Fuse after two set abstraction layers: `fusion=True`, `fuse_at_position=2`, `fuse_no_ft_pn=False`, `pointnet_pointnet=False`

### References

[1] Angela  Dai,  Angel  X.  Chang,  Manolis  Savva,  Maciej  Hal-ber, Thomas A. Funkhouser, and Matthias Nießner.  Scannet:Richly-annotated 3d reconstructions of indoor scenes. 2017.

[2] A.Dai,A.X.Chang,M.Savva,M.Halber,T.A.Funkhouser, and M. Nießner. Scannet: Richly-annotated 3d reconstruc- tions of indoor scenes. CoRR, abs/1702.04405, 2017. 1

[5] C. R. Qi, L. Yi, H. Su, and L. J. Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. CoRR, abs/1706.02413, 2017. 1
