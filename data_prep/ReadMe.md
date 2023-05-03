# Data preprocessing
This folder contains the code for the data preprocessing. Blender version 2.8 needs to be installed to run the code. 


## Dummy dataset
The 2 chairs (.dae) of the dummy dataset can be found in the `dataset` folder. To create a batch for all the chairs in the `dataset` folder and all the transformations (i.e. rotation, translation and rototranslation) with 10 views per scene, run the following command.
```
blender --background --python make_batch.py -- --n_images 50
```

The dummy dataset can be found in the `output` folder, containing 3 subfolders of the datasets of the corresponding transformations. In each of these subfolders, two scenes are stored, each containing 50 images of different views of the scene and a `render_params.json` file. There, for each view, the azimuth and elevation (in radians) of the camera and the x, y and z coordinate of the chair are stored. 

### important notes
- The camera is on a sphere with a radius of 4 (instead of 2 as the original authors did). This is because with a radius of 2, the translations were hard to capture. We could try maybe a radius of 3 if you would prefer this.
- The translations are uniformly samples every view, the positions of the chair change for the x, y and z coordinate to np.random.uniform(-0.9, 0.9).
- The camera rotates every view to an angle of np.random.uniform(0, 2*math.pi) for the x axis (elevation) and z axis (azimuth). This angle is denoted in radians. The original model expects degrees I think, so this would need to be converted. 
- The chairs are not centered exactly in middle, but on different parts, like on one of the legs. Especially, for the rotation matrix, we should probably center the object in the origin. However, they do say that the model is scene centric and not object centric so in that sense, the object should not need to be centered perfectly. Maybe this is aso why the model can already handle translations a bit?


### Things to do
- The chair objects (i.e. the .dae files) all very differ in how the chairs are defined. The location is different, how they are build (some are one object, some a collection of different objects) and the scale is very different. Moreover, the orientation and where the 'center' of the chair is differ per chair.  We have to figure out how to incoorperate all these differences.  The two chairs in the dummyset are picked such that we do not have to incoorporate these differences.

