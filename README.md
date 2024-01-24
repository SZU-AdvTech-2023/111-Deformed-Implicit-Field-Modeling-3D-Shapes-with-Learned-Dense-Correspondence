## Installation
1. Clone the repository and set up a conda environment with all dependencies as follows:
```
git clone https://github.com/microsoft/DIF-Net.git --recursive
cd DIF-Net
conda env create -f environment.yml
source activate dif
```

2. Install [torchmeta](https://github.com/tristandeleu/pytorch-meta). **Before installation, comment out [line 3 in pytorch-meta/torchmeta/datasets/utils.py](https://github.com/tristandeleu/pytorch-meta/blob/794bf82348fbdc2b68b04f5de89c38017d54ba59/torchmeta/datasets/utils.py#L3), otherwise the library cannot be imported correctly.** Then, run the following script:
```
cd pytorch-meta
python setup.py install
```


## Training a model from scratch
### Data preparation
We provide our pre-processed evaluation data from [ShapeNet-v2](https://shapenet.org/) as an example. Data can be download from this [link](https://drive.google.com/drive/folders/1VLENTGWV4VMLM1Z_9O1cAF8k9jHoB9cQ?usp=sharing) (four categories, and 100 shapes for each category respectively. 7 GB in total). The data contains surface points along with normals, and randomly sampled free space points with their SDF values. The data should be organized as the following structure:
```
DIF-Net
│
└─── datasets
    │
    └─── car
    │   │
    |   └─── surface_pts_n_normal
    |   |   |
    |   |   └─── *.mat
    │   |
    |   └─── free_space_pts
    |       |
    |       └─── *.mat    
    |
    └─── plane
    │   │
    |   └─── surface_pts_n_normal
    |   |   |
    |   |   └─── *.mat
    │   |
    |   └─── free_space_pts
    |       |
    |       └─── *.mat    
    ...
```
To generate the whole training set, we follow [mesh_to_sdf](https://github.com/marian42/mesh_to_sdf) provided by [marian42](https://github.com/marian42) to extract surface points and normals, as well as calculate SDF values for ShapeNet meshes. Please follow the instruction of the repository to install it.
### Training networks
Run the following script to train a network from scratch using the pre-processed data:
```
# train dif-net of certain category
python train.py --config=configs/train/<category>.yml
```
By default, we train the network with a batchsize of 256 for 60 epochs on 8 Tesla V100 GPUs, which takes around 4 hours. Please adjust the batchsize according to your own configuration.
## Evaluation
To evaluate the trained models, run the following script:
```
# evaluate dif-net of certain category
python evaluate.py --config=configs/eval/<category>.yml
```
The script will first embed test shapes of certain category into DIF-Net latent space, then calculate chamfer distance between embedded shapes and ground truth point clouds. We use [Pytorch3D](https://github.com/facebookresearch/pytorch3d) for chamfer distance calculation. Please follow the instruction of the repository to install it.


## Acknowledgement
This implementation takes [DIF-Net](https://github.com/microsoft/DIF-Net) as a reference. We thank the authors for their excellent work. 
