# Equivariant Neural Rendering

*Authors: Elias Dubbeldam, Aniek Eijpe, Robin Sasse, Oline Ranum, Orestis Gorgogiannis*

## Introduction


### 1.1: Intro
<!-- The paper from Dupont et al. introduces an approach to render 2D images into implicit, equivariant 3D representations. The authors argue that the scene representations need not be explicit, as long as the transformations to it occur in an equivariant manner. Their model is trained on a dataset of rotation symmetries, learning to produce novel views from a single image of a scene. -->

Current approaches in scene representations present difficulties with scalability. Voxel grids, point clouds and other traditional methods have high computational and memory requirements. Reconstrucion from incomplete or noisy data is also a challenging task with these methods, often requiring 3D information during training. Generating novel views of a scene given limited input views presents the same difficulties. Finally, traditional neural networks are not equivariant with respect to general transformation groups. 3D equivariance especially requires specifc techniques like steerable filters. Dupont et al. attempt to solve these problems by proposing a new method which results in more scalable, implicit representations that are also equivariant with respect to transformations. 

The difference between an explicit scene representation (mesh grid) and an implicit one can be seen in the figure below. While an explicit representation requires structural information of the 3D shape in great detail, the implicit representation is described by an uninterpretable three-dimensional tensor. While the numerical values of the implicit form are abstract and not meant for a human to understand, they provide significant advantages in terms of memory and computational efficiency.

![Alt text](figs/figg2.png)

To effectively use these implicit representations, the authors of the paper argue that the transformations applied to them have to be equivariant to the same transformations on an explicit representation.

  
### 1.2: Methodology
 
#### 1.2.1: Architecture

The proposed model uses a series of convolutions to map scene representations to images. Specifically, the scene representation is passed through 3D convolutions, followed by 1x1 convolutions and a set of 2D convolutions that maps them to image space. The reverse renderer is the transpose of this operation. In the figure below, an image is passed through the inverse render pipeline, creating the implicit 3D representation. Any transformation can be applied to this representation space, before following the forward renderer to obtain the recreated original, or transformed image.

Equivariance is enforced between representation and image space by applying transformations in both spaces. Because the representation space is a deep voxel grid, the transformations in this space are defined by a 3D rotation matrix. Because there is a chance that the rotated points may fall out of the reconstructed grid, inverse warping with trilinear interpolation is used in the model, to reconstruct the rotated values within the grid boundaries.

![Alt text](figs/fig5.png)

#### 1.2.1: Training

Two images of the same object, obtained from different camera angles, are passed through the inverse renderer and the implicit representations are formed. Then, the forward and backward transformation grid is applied to both latent representations to turn one into the other, before passing them to the renderer. Finally, the reconstructed output images are compared to the original inputs to obtain the loss values. Training in this manner ensures the model learns equivariant representations, as the loss evaluates both the actual rendering and the accuracy of the matching transformations from both spaces.

![Alt text](figs/fig4.png)

Finally, the authors claim that the rendering loss used makes little change in results. They provide l1 norm, l2 norm and SSIM loss as candidates, and conduct ablation studies to determine the tradeoffs between them.


### 1.3: Experiments of paper

The experiments of the study are confucted mainly on ShapeNet benchmarks, as well as two novel datasets of their design. They use an image size of 128 x 128 and a representation size of 64 x 32 x 32 x 32. The proposed model is compared against three baseline models, each one making assumptions much stronger than the original study.

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

Finally, the authors performed ablation studies to test novel view synthesis when using different loss functions. The results in each one were similar and no inherent prefered approach was suggested. In the end, they reason that choice of loss function is task specific.

## 2. Response 

Much of the success of Deep Learning can be attributed to effective representation learning. Such representations do not need to be humanly interpretable, but can also be abstract. The original authors proposed an implicit 3D representation of the scene, instead of an explicit 3D representation such as mesh-grids or point clouds. By removing the need for an explicit 3D representation, they developed a model that requires no 3D supervision. It only requires 2D images with the corresponding rotation angle of the camera, that was used between these images. Their model can generate a novel view from a single image. The qualitative results of their model’s performance motivated us to extent their research.

In the original paper the authors used 3D rotations to generate novel views, meaning that they rotate a camera on a sphere around the scene. 3D rotations do not act transitively on 3D space. Therefore, we proposed to extend their model to roto-translations, with the intermediate proof-of-concept step of using translations only. The objective was to obtain a model that can generate a novel view for any camera position in 3D space, within a reasonable range of movement.

## 3. Novel Contribution
- Describe your novel contribution.
* Methodology/theory for translation & rototranslations
  - justify group representation (homogeneous coords for translation and order of matrix multiplcation. 

- Support your contribution with actual code and experiments (hence the colab format!)
  - Demotime

  ### 3.1 Datasets

The authors present datasets consisting of rotational transformations. However, they do not provide instructions or tools for further data generation. To address this limitation we developed a new pipeline using blender for producing images of 3D-models under rotations, translations and roto-translations. Our pipeline can be used to increase the size of the training data, or to extend training data to new transformation groups.

The following section demonstrates the practical application of our pipeline for data production, enabling the generation of new training data for training translation and roto-translational invariant rendering models.

#### 3.1.1 Selecting 3D models
Similar to the authors, we perform experiments on ShapeNet benchmark. In particular, we download the [ShapeNet Core](https://shapenet.org/download/shapenetcore) subset. It is worth noting that the objects included in the ShapeNetCore dataset are already normalized and consistently aligned. From this subset we extract 2637 models.

#### 3.1.2 Build dataset with blender 

The subsequent pipeline can be adapted to accommodate any 3D-object data that is processable by Blender. Here follows a brief demonstration of how the pipeline can be used using blender 2.8. 

## 4. Conclusion

- Some preliminary results (working model)

## 5. Contributions 

Close the notebook with a description of the each students' contribution.
