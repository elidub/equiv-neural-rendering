# Data preprocessing
This folder contains the code for the data preprocessing. Blender version 3.5 needs to be installed to run the code, as instructed in the [main ReadMe](../../ReadMe.md).

- The `demo` folder contains files needed for the data demo (used by `demos/EquivariantNR.ipynb`)
- The `test_data` folder contains the test data used in the demo.
- `remove_patches.py` is a file which removes datapoints (pictures with info) from certain configurations from the training set and validation set. This is needed to create test data containing novel views (i.e. views from a scene the model has not seen during training).
- `render_blender.py` is the file that creates scene images from a given object. When running this file, the path to blender has to be specified first. Moreover, it expects and can handle multiple arguments, namely;
    - `--scene_name`:  the name of the scene folder, in which the `.obj` file, obtained from ShapeNet, is specified. 
    - `--scene_folder`: the path to the above mentioned scene name.
    - `--n_images`: number of views to be rendered for a given scene.
    - `--output_folder`: directory the created data will be stored in.
    - `--resolution`: defines the resolution of the images taken, default = 64.
    - `--rotation`: if this argument is specified, the images will differ in camera angle.
    - `--translation`: if this argument is specified, the images will differ in object location.

For example, `render_blender.py` can be run with the following commandline to create 5 views with a change in rotation and in translation:
```
blender -b --python render_blender.py -- --scene_name demo/data/model_1 --n_images 5 --output_folder demo/output --rotation --translation
```