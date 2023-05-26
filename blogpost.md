# Equivariant Neural Rendering

*Authors: Elias Dubbeldam, Aniek Eijpe, Robin Sasse, Oline Ranum, Orestis Gorgogiannis*

## Introduction

<!-- The paper from Dupont et al. introduces an approach to render 2D images into implicit, equivariant 3D representations. The authors argue that the scene representations need not be explicit, as long as the transformations to it occur in an equivariant manner. Their model is trained on a dataset of rotation symmetries, learning to produce novel views from a single image of a scene. -->

Current approaches in scene representations present difficulties with scalability. Voxel grids, point clouds and other traditional methods have high computational and memory requirements. Reconstrucion from incomplete or noisy data is also a challenging task with these methods, often requiring 3D information during training. Generating novel views of a scene given limited input views presents the same difficulties. Finally, traditional neural networks are not equivariant with respect to general transformation groups. 3D equivariance especially requires specifc techniques like steerable filters. Dupont et al. attempt to solve these problems by proposing a new method which results in more scalable, implicit representations that are also equivariant with respect to transformations. 

The difference between an explicit scene representation (mesh grid) and an implicit one can be seen in the figure below. While an explicit representation requires structural information of the 3D shape in great detail, the implicit representation is described by an uninterpretable three-dimensional tensor. While the numerical values of the implicit form are abstract and not meant for a human to understand, they provide significant advantages in terms of memory and computational efficiency.

![Alt text](figs/figg2.png)

To effectively use these implicit representations, the authors of the paper argue that the transformations applied to them have to be equivariant to the same transformations on an explicit representation.

  
### 1.1: Methodology
 
#### 1.1.1: Architecture

The proposed model uses a series of convolutions to map scene representations to images. Specifically, the scene representation is passed through 3D convolutions, followed by 1x1 convolutions and a set of 2D convolutions that maps them to image space. The reverse renderer is the transpose of this operation. In the figure below, an image is passed through the inverse render pipeline, creating the implicit 3D representation. Any transformation can be applied to this representation space, before following the forward renderer to obtain the recreated original, or transformed image.

Equivariance is enforced between representation and image space by applying transformations in both spaces. Because the representation space is a deep voxel grid, the transformations in this space are defined by a 3D rotation matrix. Because there is a chance that the rotated points may fall out of the reconstructed grid, inverse warping with trilinear interpolation is used in the model, to reconstruct the rotated values within the grid boundaries.

![Alt text](figs/fig5.png)

#### 1.1.1: Training

Two images of the same object, obtained from different camera angles, are passed through the inverse renderer and the implicit representations are formed. Then, the forward and backward transformation grid is applied to both latent representations to turn one into the other, before passing them to the renderer. Finally, the reconstructed output images are compared to the original inputs to obtain the loss values. Training in this manner ensures the model learns equivariant representations, as the loss evaluates both the actual rendering and the accuracy of the matching transformations from both spaces.

![Alt text](figs/fig4.png)

Finally, the authors claim that the rendering loss used makes little change in results. They provide l1 norm, l2 norm and SSIM loss as candidates, and conduct ablation studies to determine the tradeoffs between them.

### 1.2: Datasets
The authors evaluate their model on 4 datasets, including two ShapeNet benchmarks as well as two novel datasets of the authors design. They use an image size of 128 x 128 and a representation size of 64 x 32 x 32 x 32.
The datasets are presented in table 1. 


| *Dataset*  | *Source*  |  *Sample* | *# Scenes*  |*# images per scene*| *# datapoints*|
|---|---|---|---|---|---|
| Chairs  | [ShapeNet](https://icml20-prod.cdn-apple.com/eqn-data/data/chairs.zip)  | ![Chair](./src/imgs/paper_screenshots/chair.png)  |  6591 | 50  | 329 550|
| Cars  |  [ShapeNet](https://icml20-prod.cdn-apple.com/eqn-data/data/cars.zip) | ![Car](./src/imgs/paper_screenshots/car.png)  |  3514 |  50 | 175 700|
| MugsHQ  |  [Apple](https://icml20-prod.cdn-apple.com/eqn-data/data/mugs.zip) | ![Mug](./src/imgs/paper_screenshots/mug.png)  |  214 | 150  | 32 100|
| 3D mountainset  |  [Apple](https://icml20-prod.cdn-apple.com/eqn-data/data/mountains.zip) | ![Mountain](./src/imgs/paper_screenshots/mountain.png)  |  559 |  50 | 27 950|

Table 1.: *Overview of datasets considered for equivariant neural rendering by Dupont et al.*


### 1.3: Experiments of paper

The experiments of the study are confucted mainly on ShapeNet benchmarks, as well as two novel datasets of their design. They use an image size of 128 x 128 and a representation size of 64 x 32 x 32 x 32. The proposed model is compared against three baseline models. All three built for 3D rendering from one or multiple 2D images, but they all make assumptions much stronger than the original study. 

|   | TCO  |  DGQN | SRN  | Proposed model  |
|---|---|---|---|---|
| Requires Absolute Pose  | Yes  | Yes | Yes | No |
| Requires Pose at Inference Time  | No  | Yes | Yes | No |
| Optimization at Inference Time  | No  | No | Yes | No |

The qualitative comparisons against the baseline models in single shot novel view synthesis with the ShapeNet chairs dataset reveals that the model achieves similar to SoTA results while making far fewer assumptions than the other methods. It can produce high quality novel views by achieving the desired equivariant transformation in representation space.

![Alt text](figs/results.png)

Experiments in other datasets include:

- Cars: the cars Shapenet class
- MugsHQ: a dataset of mugs based on the mugs ShapeNet class with an added background environment
- 3D mountains: a dataset of mountain landscapes

Results similar to the chairs were reported in the other datasets, with some variations due to the specific challenges of each one. For example, the mountains contain extremely complex geometric information, which severly limits the detail of the novel view synthesis.

![Alt text](figs/chairs.png) 

![Alt text](figs/cars.png) 

![Alt text](figs/mugs.png) 

![Alt text](figs/mountains.png)

Finally, the authors performed ablation studies to test novel view synthesis when using different loss functions. The results in each one were similar and no inherent prefered approach was suggested. In the end, they reason that choice of loss function is task specific. Their claim is supported by their experiments with different losses, showing minimal qualitative images in outputs.

### Loading, plotting & transforming the original image

The model infers from a single image and renders a second image from a novel view, as illustrated in the figures below.

![Alt text](figs/demo1.png) 

![Alt text](figs/demo2.png)


## 2. Response 

Much of the success of Deep Learning can be attributed to effective representation learning. Such representations do not need to be humanly interpretable, but can also be abstract. The original authors proposed an implicit 3D representation of the scene, instead of an explicit 3D representation such as mesh-grids or point clouds. By removing the need for an explicit 3D representation, they developed a model that requires no 3D supervision. It only requires 2D images with the corresponding rotation angle of the camera, that was used between these images. Their model can generate a novel view from a different angle, given a single image. The qualitative results of their model’s performance motivated us to extent their research.

In the original paper the authors used 3D rotations to generate novel views, meaning that they rotate a camera on a sphere around the scene. 3D rotations do not act transitively on 3D space. Therefore, we proposed to extend their model to roto-translations, with the intermediate proof-of-concept step of using translations only. Roto-translations act transitively on 3D space, meaning that we can produce any possible camera angle. The full spectrum of rigid body motions are nescessary to produce satisfactory renderings of real world environments. For instance in applications such as virtual and agumented reality, and 3D reconstruction. Hence we hoped to obtain a model that can generate a novel view for any camera position in 3D space, within a reasonable range of movement.


###  What can the model do and what is missing?

The model, which was trained by the original authors, shows some nice out-of-the-box capabilities. That is the model can already perform some limited (roto-)translations.

#### Translations through inductive bias

Through inductive bias, (reasonably small) translations, which are orthogonal to the line of sight, already work on the model that has only been trained on rotations. This is due to the fact that the model uses a CNN architecture, which is translationally equivariant along the image plane. Due to the weight sharing property of the convolution kernels, a CNN will generally use the same encoding for the same image but with the same shift applied to the encoding. The same goes for the decoder (i.e. renderer) which will produce the same rendered image for a shifted reprensentation. Therefore, the model acts translationally equivariant for these kinds of shifts. Still, it seems interesting that the model does not seem to encode any information from the outside of the object to produce a good estimation.

![Alt text](./src/imgs/paper_screenshots/translational_eq_cnn.png)

*Illustration of translational equivariance in CNNs (https://www.mdpi.com/2076-3417/10/9/3161). Shifting the input and encoding it is equivalent to shifting encoding the input and shifting the encoding.*

Nonetheless, translations along the line of sight do not work out-of-the-box and require explicit training. The reason for that is that the equivariant neural rendering model considers the depth dimension via incorporating its information into the channels of the CNN. More concrete, the model uses the following code:

```python
# Reshape 3D -> 2D
reshaped = inputs.view(batch_size, channels * depth, height, width)
```

Furthermore, due to the central positioning of the objects in the images, the model has problems rendering scenes that extent to the image boundaries. Therefore, we trained our model on translations first, before moving on to roto-translations

#### Translations

![Alt text](figs/translations.png)

Another problem with out-of-the-box translations from the rotation model is that it only shifts the 2D image instead of developing a real 3D understanding of the scene. When we compare the rendered image to the ground truth, we observe that the model does not grasp that a shift also changes the angle at which the camera is looking at the object. It is obvious because the model has never seen a shift and only works on 2D equivariance as described above. Furthermore, we simply added a functionality for translations without ever training the model on them.

![Alt text](figs/translations2.png)

#### Roto-translations

We further observe that the same properties allow for out-of-the-box roto-translations. Also the roto-translations do not account for the angular shift between camera and object.

![Alt text](figs/rototrans.png)

#### In conclusion...

Their model has some useful capabilities for generating novel views including (roto-)translations. Nonetheless, upon thorough review, these translations do not reflect actual physics (relative rotational angle), nor are they complete (zoom and edge artifacts). We therefore need to train a model which can produce these novel views correctly.


## 3. Novel Contribution

In this section we describe the novel contributions of our research.

- We introduce a method to generate training data for the equivariant neural rendering models (section 3.1).

- We introduce a model that has been trained on translations and a model that has been trained on roto-translations (section 3.2). This part constitutes the main contribution of our research.

### 3.1 Datasets

The authors present datasets consisting of rotational transformations. However, they do not provide instructions or tools for further data generation. To address this limitation we developed a new pipeline using blender for producing images of 3D-models under rotations, translations and roto-translations. Our pipeline can be used to increase the size of the training data, or to extend training data to new transformation groups.

The following section demonstrates the practical application of our pipeline for data production, by demonstrating how to use blender to generate new training data containing roto-translations.

#### 3.1.1  Demonstration: populating datasets for the ISO(3)-group using Blender 
Similar to Dupont et al., we perform experiments on the [ShapeNet Core](https://shapenet.org/download/shapenetcore)-Chairs benchmark. It is worth noting that the objects included in the ShapeNetCore dataset are already normalized and consistently aligned. However, the subsequent pipeline can be adapted to accommodate any 3D-object data that is processable by Blender. Here follows a brief demonstration of how data can be constructed using Blender 3.5.1.



#### 3.1.2 Populating new datasets

We use the afformention pipeline to build 3 new datasets: 

   * _Rotations_: used to reproduce the results presented by Dupont et al.
   * _Translations_: used to train a model with higher capacity for translation invariance.
   * _Roto-translations_: used to train a roto-translational invariant model.
    
We downscale the datasets in order to reduce the computational costs of training the new models. For all three datasets we use the partitioning described in table 2. 


|   | **# Scenes**  |  **# Images per scene** | **Resolution**  | **# datapoints**  |
|---|---|---|---|---|
| Train  | 2306  |  50 | 64 x 64  |  115300 |
| Validataion  | 331  | 50  |  64 x 64 | 16550  |

Table 2: _Partition of new datasets_


| **Hyperparameter**  | **R**  |  **X** | **Y**  | **Z**  | **Resolution** |
|---|---|---|---|---|---|
|   | 1.5  | [-0.4, 0.4]  | [-0.3, 0.5]  | [-0.4, 0.4] | 64 x 64|

Table 3: _Hyperparameters used when populating the new dataset._

% Need to include rotations for training here, and fill in more info.

We construct the datasets by sampling poses from various views. In case of rotations the camera is placed on a sphere with a radius **R**. For each view, a value between 0 and $2\pi$ is uniformly sampled for the elevation and azimuth angle of the camera and rotated accordingly. In case of translations, for each view, a value is uniformly sampled from a range of **X**, **Y** and **Z** locations of the chair.


### 3.2 Extending the model

After evaluating the pretrained model supplied by Dupont et al., we extend the architecture to allow for translation matrices to be applied to the input image. Furthermore, by combining it with the previously implemented rotation matrix, we also allow for rototranslations. 

**TODO: WRITE MORE ABOUT MODEL ARCHITECTURE EXTENSIONS**

With these extentions to the model architecture, we conducted the following experiments:


#### 3.2.1 Reproducing rotations with our dataset

With the dataset we created as discussed in 3.1, we tried training a rotation-based model from scratch, hoping to reproduce the authors' original results on our data. The results were 

![Alt text](figs/ourRot1.png)

![Alt text](figs/ourRot2.png)

![image](./src/imgs/output/rotations.gif)




## 4. Conclusion

- Some preliminary results (working model)

## 5. Contributions 

Close the notebook with a description of the each students' contribution.
