# SYNTHIA_FPNet
Frustum PointNet for 3D Object Detection from Point Clouds using SYNTHIA Dataset


## STEPS


### STEP 0. Convert SYNTHIA to KITTI on server


0.1 . Change directory structures in ````code/new_synthia_to_kitti.py```` as follows:

````
# Changes in data dir
# Read Operations --> change according to need

data_dir = '/../../datatmp/Datasets/detection/SynthiaReloaded/50_model_illumination/'

# Write Operations --> change according to need
  
IMG_ROOT = '/../../datatmp/Datasets/detection/SynthiaReloaded/SYNTHIA_KITTI_1/object/training/image_2/'

PC_ROOT = '/../../datatmp/Datasets/detection/SynthiaReloaded/SYNTHIA_KITTI_1/object/training/velodyne/'

CALIB_ROOT = '/../../datatmp/Datasets/detection/SynthiaReloaded/SYNTHIA_KITTI_1/object/training/calib/'

LABEL_ROOT = '/../../datatmp/Datasets/detection/SynthiaReloaded/SYNTHIA_KITTI_1/object/training/label_2/'

# Comment following if not needed
ANOT_IMG_ROOT = '/../../datatmp/Datasets/detection/SynthiaReloaded/SYNTHIA_KITTI_1/object/training/anot_image_2/'
````

0.2. Run following from code:
````
python3 new_synthia_to_kitti.py
````


### STEP 1. Generate train pickle locally


NOTE: All the following changes (for 1.) are for `local/syn_fpnet`

1.1.  Copy data from server (`SynthiaReloaded/SYNTHIA_KITTI_1`) to local machine in folder syn_fpnet/dataset (and rename to KITTI) 

1.2. Change (write) image_sets/train.txt to required number of training samples (script provided in `SYNTHIA_KITTI_conversion/gen_traintxt.py`) 

1.3. Change script command_prep_data.sh as follows to generate only train pickle:
````
python kitti/prepare_data.py --gen_train 
````
1.4. Change kitti/kitti_object.py line number 34 to required number of training samples as follows:
````
self.num_samples = 8088
````
1.5. Then to prepare the data, simply run: 
````
sh scripts/command_prep_data.sh
````

### STEP 2. 
2.1 Copy val pickle files from (previous work) server/new_fpnet/kitti folder to server/syn_fpnet/kitti

### STEP 3. Train on server

3.1. Open screen in server
3.2. Make the following changes for GPU number to scripts/command_train_v1.sh:
````
python train/train.py --gpu 4 --model frustum_pointnets_v1 --log_dir train/log_v1 --num_point 1024 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5
````
3.3. Then run the following:
````
CUDA_VISIBLE_DEVICES=4 sh scripts/command_train_v1.sh
````

### Visualization of Point Clouds with bounding boxes


Run the following from syn_fpnet:
````
python kitti/prepare_data.py --demo
````



