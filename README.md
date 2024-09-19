## Point cloud compression with bits-back coding
This repository is the code for reproducing results in our paper 
[Point Cloud Compression with Bits-back Coding](https://hieunq95.github.io/).

### Dependency
We will need to install `numpy`, `autograd`, `torch`, `matplotlib`, and `open3d`.

`pip install -r requirements.txt`

### 1. Preparing datasets
We will use [ShapeNet](https://huggingface.co/datasets/ShapeNet/ShapeNetCore) 
and [SUN RGB-D](https://rgbd.cs.princeton.edu/challenge.html) as two datasets for 
training a convolutional variational autoencoder (CVAE).

For ShapeNet dataset, we will create a customized dataset of 5 object classes from ShapenetCore that are 
`04379243` (table), `02958343` (car), `03001627` (chair), `02691156` (airplane), and `04256520` (04256520).

Let's download the corresponding zip files from the ShapeNetCore repository 
[here](https://huggingface.co/datasets/ShapeNet/ShapeNetCore/tree/main) on HuggingFace. For example, just search
`04379243.zip` in the HuggingFace repository and download the zip file.

For SUN RGB-D dataset, we just download the zip files `SUNRGBD.zip` and `SUNRGBDLSUNTest.zip` from the website 
for training and testing sets, respectively.

Once download the zip files, let's unzip the files and place them in the same folder named `~/open3d_data/extract/`.

To make the folders, use following commands

`mkdir ~/open3d_data`  
`cd open3d_data`  
`mkdir extract`  
`mkdir ShapeNet`  
`mkdir processed_shapenet`  
`mkdir processed_sunrgbd`

The folders `ShapeNet` will contain the extracted mesh objects from the ShapeNet dataset, 
folders `processed_shapenet` and `processed_sunrgbd` will contain processed train set and test set (in `npy` format) 
for training the CVAE model.

Let's extract all file ShapeNet zip files `04379243.zip`, `02958343.zip`, etc, into the newly created folder `ShapeNet`.
Then we extract `SUNRGBD.zip` and `SUNRGBDLSUNTest.zip` files into the folder `extract`.
Once the files are extracted, we have the folders look like this:

hieu/  
|--- open3d_data/  
|--------- extract/  
|---------------- ShapeNet/  
|---------------------------- 04379243/  
|---------------------------- 02958343/  
|---------------------------- 03001627/  
|---------------------------- 02691156/  
|---------------------------- 04256520/  
|---------------- SUNRGBD/  
|---------------- SUNRGBDv2Test/  
|---------------- processed_shapenet/  
|---------------- processed_sunrgbd/

### 2. Creating customized datasets
#### 2.1 Creating training sets
Now let's creating the customized datasets as described in the paper.
The customized ShapeNet training sets can be created by running the following commands, one by one:

`python dataset.py ---make 1 --mpc 2000 --res 32 --mode train --type shape`  
`python dataset.py ---make 1 --mpc 2000 --res 64 --mode train --type shape`  
`python dataset.py ---make 1 --mpc 2000 --res 128 --mode train --type shape`

**Note: make sure to type the commands by yourself rather than copying from the text above. This is 
to avoid the error `error: unrecognized arguments: ---make 1` (due to confusion between str and int format of python).

The command above just created three voxelized point cloud datasets for training. 
The `--res` parameter controls the resolution (bit-depth value `d` described in the paper).

After running each command (take around 1 hour or less for each command), we will have three
new files in the `processed_shapenet` folder named `shapenet_train_32.npy` (261 MB), `shapenet_train_64.npy` (2.1 GB), 
and `shapenet_train_128.npy` (16.8 GB). 

Similarly, customized SUN RGB-D training sets can be created by running the following commands, one by one:

`python dataset.py ---make 1 --res 32 --mode train --type sun`  
`python dataset.py ---make 1 --res 64 --mode train --type sun`  
`python dataset.py ---make 1 --res 128 --mode train --type sun`

We will have three new files `sunrgbd_train_32.npy` (338 MB), `sunrgbd_train_64.npy` (2.7 GB), 
and `sunrgbd_train_128.npy` (21.7 GB) 
in the `processed_sunrgbd` folder.

#### 2.2 Creating testing sets
Let's create ShapeNet test set and SUN RGB-D test as by running  

`python dataset.py ---make 1 --mpc 2000  --mode test --type shape`  
`python dataset.py ---make 1 --mode test --type sun`  

Note that we don't need the `--res` parameter in the input as the test sets are raw point cloud data (not voxelized).

### 3. Training CVAE model
To train the CVAE model on ShapeNet training set with bit-depth `d=6` for 500 epochs, we run the following command  

`python main.py --mode train --ep 500 --res 64 --type shape`.  

After running the above command, we will obtain a trained model named `params_shape_res_64` in the
[model_params](https://github.com/hieunq95/gpcc-bits-back/tree/main/model_params)
folder. This file will be used later to load the parameters of the CVAE at the testing phase.

For training the model on other training sets, we can change the `--res` and `--type` parameters.

A set of pre-trained models is already in the
[model_params](https://github.com/hieunq95/gpcc-bits-back/tree/main/model_params) folder for reference.

### 4. Testing CVAE model
To reproduce the visualization in the paper, we can run the command

`python main.py --mode test --res 64 --type shape`.

Similarly, we can vary the parameters `--res` and `--type` for obtaining other visualization results.

### 5. Compressing voxelized point clouds with CVAE model

Before recreating some figures in the paper, let's install [Draco](https://github.com/google/draco) as we will later use
it as a baseline. For this, make sure to follow instructions from the 
Draco' github [repo](https://github.com/google/draco/blob/main/BUILDING.md).

After installing Draco as a software, let's copy the following files (appear in the `build_dir` 
based on the [instruction](build_dir)) into our 
[draco](https://github.com/hieunq95/gpcc-bits-back/tree/main/draco) folder:

`draco draco.pc  draco_decoder draco_decoder-1.5.7 draco_encoder draco_encoder-1.5.7`.

By doing so, we will later use python wraper functions to use Draco to compress some point cloud data.

Finally, let's compress some voxelized point cloud data with our pre-trained CVAE model:  

`python main.py --mode eval_rate --res 64 --type shape`

This will create Fig. 5 in the paper. To redraw the figure, we can run 

`python main.py --mode plot_rate`

To reproduce Fig. 6 in the paper, we run the following commands

`python main.py --mode eval_depth --type shape --batch 999`  
`python main.py --mode eval_depth --type sun --batch 999`

To redraw the figure, we run the command

`python main.py --mode plot_depth`.


### Acknowledgement
Many thanks to Craystack's contributors (Jamie, Daniel, and others) that make the project publicly available.  
A part of code from this paper is developed based on the [Craystack](https://github.com/j-towns/craystack) 
code repository.
