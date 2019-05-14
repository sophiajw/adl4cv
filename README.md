# adl4cv
Advances Deep Learning for Computer Vision

Our project:

## Scene Completion and Semantic Segmentation in Point Clouds

1. Introduction
In this project, we will investigate, whether it is benefi- cial for semantic scene segmentation to firstly complete the point cloud. We will use PointNet++ [5] as a basis and from there try to improve it’s performance using a modified ver- sion of Point Completion Network [6].
2. Related Work
There are already many recent research works on scene completion and semantic segmentation tasks on 3D data. Often, the underlying data come from RGB-D scans and are processed to voxels. This approach is handy for the application of 3D-convolutions and allows a direct evalu- ation of a learning method’s performance as it works on a regular grid [3]. However, computation and training of voxel-based methods are very costly. This led to the explo- ration of novel methods on more efficient representations such as point clouds. Important advances in the direction of semantic segmentation have been the deep learning ar- chitecture PointNet [4] and its extension PointNet++ that includes metric space distances to learn also local features. So far, completion tasks on point clouds have only been re- alized on single shapes, for example with PCN, but not on scenes.
3. Method
3.1. Contributions
In contrast to the other methods that focus on completing point clouds for single shapes, we would like to implement a network structure that is able to complete point clouds for entire scenes. Afterwards, we would like to evaluate the completion not only by some metrics, but also on how much it improves the semantic segmentation on the scenes.
3.2. Methodology
Our first step is to implement PointNet++ for semantic scene segmentation. The performance of this network will be our benchmark for the evaluation of the improvement when using a completed point cloud. As evaluation metric, we will use the mIoU metric.
For the completion, we will start with implementing PCN, which consists on a feature extraction path, followed by an upsampling path to predict the resulting shape. After that, we want to see which parts of the network have to be adjusted for scene completion.
The usual PCN encoding path consists of two consec- utive PointNet layers. One option is to refine the encoder similarly to PointNet++, and apply a cascade of PointNet layers in order to account for the different levels of coarse- ness in the scene.
Another possibility is applying the whole algorithm re- peatedly using different coarseness levels as input together with the output of the previous coarseness level (as seen in the voxelized approach of ScanComplete).
The performance of the network will be measured using a combination of Earth Mover’s Distance and Chamfer Dis- tance.
4. Datasets and Resources
For our task we need a dataset that is suited for both semantic segmentation and scene completion and consists of entire scenes instead of single shapes. We choose the dataset ScanNet [2], consisting of semantic annotated 3D- indoor scenes. Though the scans are not complete, we need to generate incomplete training data ourselves by sampling points that we omit. By this, we can use the actual dataset as ground truth for completion. Another well suited candi- date, the SemanticKITTI dataset [1], is only released in late summer 2019 and could be used in further studies.
We have a Nvidia GPU of 2GB. This should be sufficient since computations on point clouds are much more efficient compared to voxelized approaches. As a backup solution we would use the provider Collab.
5. Milestones
We would like to set up the PointNet++ model before the first presentation. Our next step, the completion network, should be working before the second presentation. After that, we will combine the two networks to examine whether the performance of PointNet++ is improved by our comple- tion. If we still have time left, we would like to investigate if segmentation improves completion vice versa.

References
[1] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stachniss, and J. Gall. A dataset for semantic segmentation of point cloud sequences. CoRR, abs/1904.01416, 2019. 1
[2] A.Dai,A.X.Chang,M.Savva,M.Halber,T.A.Funkhouser, and M. Nießner. Scannet: Richly-annotated 3d reconstruc- tions of indoor scenes. CoRR, abs/1702.04405, 2017. 1
[3] A. Dai, D. Ritchie, M. Bokeloh, S. Reed, J. Sturm, and M. Nießner. Scancomplete: Large-scale scene completion and semantic segmentation for 3d scans. In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 2018. 1
[4] C. R. Qi, H. Su, K. Mo, and L. J. Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. CoRR, abs/1612.00593, 2016. 1
[5] C. R. Qi, L. Yi, H. Su, and L. J. Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. CoRR, abs/1706.02413, 2017. 1
[6] W. Yuan, T. Khot, D. Held, C. Mertz, and M. Hebert. PCN: point completion network. CoRR, abs/1808.00671, 2018. 1
