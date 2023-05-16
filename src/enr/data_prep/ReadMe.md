# Data preprocessing
This folder contains the code for the data preprocessing. Blender version 2.8 needs to be installed to run the code. 

## Dummy dataset
The dummy dataset can be found in the `output_dummy` folder, containing 3 subfolders of the datasets of the corresponding transformations. In each of these subfolders, 10 scenes are stored, each containing 50 images of different views of the scene and a `render_params.json` file. There, for each view, the azimuth and elevation (in radians) of the camera and the x, y and z coordinate of the chair are stored. 

## Install Blender on LISA
- Download Blender for Linux manually from the [webiste](https://www.blender.org/download/), or [here](https://www.blender.org/download/release/Blender3.5/blender-3.5.1-linux-x64.tar.xz/) (248MB) directly. Should also be able to do this directly on LISA with `wget`, but haven't been able to figure that out.
- Copy the file `blender-3.5.1-linux-x64.tar.xz` to the director `/equiv-neural-rendering`.

```
# Unpack 
tar -xvf blender-3.5.1-linux-x64.tar.xz

# Rename for shorter commands
mv blender-3.5.1-linux-x64 blender
rm blender-3.5.1-linux-x64.tar.xz

# Go to a GPU node and activate conda env (see below)
# Not sure if conda env is even necessary

# Go to data_prep and run an example blender command
cd data_prep
~/equiv-neural-rendering/blender/blender -b --python render_blender.py -- --scene_name dataset/model.dae --rotation
# Thist last command is just an example for now, I guess one can also run `make_batch.py` with some changes.
```
> I have renamed `"View Layer"` to `"ViewLayer"` in `render_blender.py` (see commits), so it might not work anymore locally!?


### Activate GPU node 
```
srun --partition=gpu_titanrtx_shared_course --gres=gpu:1 --mem=32000M --ntasks=1 --cpus-per-task=3 --time=01:00:00 --pty bash -i

module purge
module load 2021
module load Anaconda3/2021.05
conda activate nr
```
