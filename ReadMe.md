# Equivariant Neural Rendering

*Authors: Elias Dubbeldam, Aniek Eijpe, Robin Sasse, Oline Ranum, Orestis Gorgogiannis*

This repository contains code and blogpost on the reproduction and extension of [Equivariant Neural Rendering](https://arxiv.org/abs/2006.07630), ICML 2020. We present framework for learning neural scene implicit scene representations directly from images. The framework is able to render a scene from a single image. We present models trained on rotations-only, translations-only and roto-translations. For an in-depth discussion of our work, see the blogpost.


## Code structure

| Directory | Description |
| --------- | ----------- |
| `demos/` | Notebooks used to analyze training runs, results and render scenes. |
| `src/configs/` | Configuration files to run the training experiments. |
| `src/enr` | Source code used for equivariant neural rendering. Adapted from the [repo of the original paper](https://github.com/apple/ml-equivariant-neural-rendering). |
| `src/imgs/` | Location where produced images are stored. |
| `src/scripts/` | Files to run that reproduce the results. |
| `src/train_results/` | Location where all trained models and its (intermediate) results are stored. |
| `blogpost.md` | Report introducing original work, discussing our novel contribution and analaysis. |



## Installation & download

<!-- To create the necessary data, follow the instructions in the [ReadMe about the data](src/enr/data/ReadMe.md) -->

First, install the conda environment and [Blender](https://www.blender.org/)
```shell
conda env create -f env_nr.yml
conda activate nr

# Download blender 3.5.1
!wget https://ftp.nluug.nl/pub/graphics/blender/release/Blender3.5/blender-3.5.1-linux-x64.tar.xz

# Unpack 
!tar -xvf blender-3.5.1-linux-x64.tar.xz
!rm ./blender-3.5.1-linux-x64.tar.xz

# Move and rename for shorter commands
!mv ./blender-3.5.1-linux-x64 ./src/enr/data/demo/blender
```

Finally, download the [ShapeNet Core dataset](https://shapenet.org/login/), the subset comprised of chair models can be found in folder *03001627*.

## Usage

#### Creating data

After the installation insctructions from above, one can run the scripts in `src/scripts/` to create the data and reproduce the training runs and results. To create the necessary data, run the following:

```shell
python src/enr/data/create_data.py --blender_dir <blender location> --obj_dir <location of blender objects to read> \\
--output_dir <location to store the data> --transformation <rotation, translation or roto-translation>
```
Depening on the `transformation`-flag, one produces images from the objects in `obj_dir` with Blender See the [ReadMe about the data](src/enr/data/ReadMe.md) for more information.

#### Training a model
To train a model, run the following:

```shell
python src/scripts/experiments.py config.json
```
See the configuration files in `src/configs/` for detailed training options, such as training on rotations,translations or roto-translations, single or multi-GPU and more. To reproduce our results, use the `config.json` files as they are saved in `/src/train_results/<timestamp>_<id>`. That is, one should change `id`, `path_to_data` and `path_to_test_data` accordingly, to the paths where one created the data following the instructions in the [ReadMe about the data](src/enr/data/ReadMe.md). 

#### Evaluate results and produce renderings

To evaluate a model, run the following:

```shell
python src/scripts/evaluate_psnr.py <path_to_model> <path_to_data>
```
By default, models are stored in `src/train_results/`, The path to data is dependent on the user, where it has saved its data. See the configuration files in `src/configs/` for more information.

One can create gifs that render the scene from different viewpoints by running the following:

```python
python src/scripts/render_gif.py --img_path <source_img_path> --model_path <model_path>
```
Models are stored in `src/train_results/<timestamp>_<id>/best_model.pt`, The path to data is dependent on the user, where it has saved its data.

#### Analyze results and explore the model

Finally, the model can be analyzed and used to render scenes using the notebooks in `demos/`. The notebook `demos/analyze_training.ipynb` shows the loss curves of the different training runs. The notebook `EquivariantNR.ipynb` shows how to use a trained model to infer a scene representation from a single image and how to use this representation to render novel views.

An in-depth discussion and analysis on the results can be found in the blogpost.

## License
As this code has been built upon the the [repository `apple/ml-equivariant-neural-rendering`](https://github.com/apple/ml-equivariant-neural-rendering), the same license applies. That is, this project is licensed under the Apple Sample Code License.
