import random, os, sys
import cv2
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import image as mpimg
import pprint
import imageio
import torch
import torchvision
from torchvision.transforms import ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" Detect local path """
local_path = os.getcwd()

main_path = local_path[:-5]


sys.path.append(local_path + '/../src/')
sys.path.append(local_path + '/../src/enr/')
from models.neural_renderer import *

# ---------------------------------------------------------
# Configure transformations for demonstration
# ---------------------------------------------------------

def transformation_dict(azimuth = 0., elevation = 0., x = 0., y = 0., z = 0., print_transform = False):
    """ Store transformations in dictionary
    
    Args:
        azimuth (angle): Positive (negative) values correspond to moving camera to the right (left)
        Elevation (angle): Positive (negative) values correspond to moving camera up (down)
        x, y, z (scalar units): translations along x, y, z axis 
    """
    transformations = {}
    
    # Rotation values
    transformations['azimuth'] = torch.Tensor([azimuth]).to(device)
    transformations['elevation'] = torch.Tensor([elevation]).to(device)

    # Translation values
    transformations['xyz'] = torch.Tensor([x, y, z]).to(device)

    if print_transform:
        pprint.pprint(transformations)
    return transformations

def bprint(str):
    return print("\033[1m" + '\n' + str + '\n' + "\033[0;0m")

def tensor_to_image(img, nrow=4):
    """Helper function to plot image tensors.
    
    Args:
        img (torch.Tensor): Image or batch of images of shape 
            (batch_size, channels, height, width).
    """
    
    img_grid = torchvision.utils.make_grid(img, nrow=nrow)
    return img_grid.cpu().numpy().transpose(1, 2, 0)

def render_image(model_path, img_path, downsample = False):
    # ---------------------------------------------------------
    # Loading the original image
    # ---------------------------------------------------------
    # Load trained chairs modellocal_path
    model = load_model(model_path).to(device)

    # You can also try loading other examples (e.g. 'chair1.png')
    original_img = plt.imread(img_path)
    
    if downsample:
        # Convert image to tensor and add batch dimension
        original_img = cv2.resize(original_img, dsize=(64, 64), interpolation=cv2.INTER_AREA)
        
    img_source = ToTensor()(original_img)
    img_source = img_source.unsqueeze(0).to(device)
    
    if img_source.shape[2] != 3:
        img_source = img_source[:, 0:3, : , :]
        
    # ---------------------------------------------------------
    # Render original image without transformations
    # ---------------------------------------------------------

    # Infer scene representation
    scene = model.inverse_render(img_source)

    # We can render the scene representation without transforming it
    rendered = model.render(scene)
    rendered_img = tensor_to_image(rendered.detach())
    
    return scene, model, original_img, rendered_img

def transform_scene(model,scene, init_camera_pos, transformations):
    """Helper function to transform scene
    
    Args: 
        model: model used to render scene
        scene: implicit scene representation
        transformations (dict): dict comprised of keys
            azimuth (angle): Positive (negative) values correspond to moving camera to the right (left)
            elevation (angle): Positive (negative) values correspond to moving camera up (down)
            Translations (dict): translation values with dictionary elements corresponding to 'x', 'y', 'z'
            
     As a rotation matrix can feel a little abstract, we can also reason in terms of 
         camera azimuth and elevation. The initial coordinate at which the source image
         is observed is given by the following azimuth and elevation. Note that these
         are not necessary to generate novel views (as shown above), we just use them 
         for convenience to generate rotation matrices
    """
    # Set Transformations and Configure output view
    azimuth_target = init_camera_pos['azimuth'] + transformations['azimuth']
    elevation_target = init_camera_pos['elevation'] + transformations['elevation']
    translations_target = init_camera_pos['xyz'] + transformations['xyz']

    # Rotate scene to match target camera angle
    transformed_scene = model.rotate_source_to_target(
        scene, 
        init_camera_pos['azimuth'], init_camera_pos['elevation'], init_camera_pos['xyz'],
        azimuth_target, elevation_target, translations_target)
    
    # Render rotated scene
    rendered = model.render(transformed_scene)
    image = tensor_to_image(rendered.detach())
    
    return image

def display_transformations(imgs, keys, fsize = (8, 6)):
    """ Helper function to make suplot displaying images
    
    Args:
        imgs (list): list of images to display
        keys (list): list of plot titles
        fsize (2-touple): figure size
    """
    N = len(imgs)
    fig, axs = plt.subplots(1, N, figsize=fsize)
    for ax, idx in zip(axs, range(N)):
        ax.imshow(imgs[idx])
        ax.title.set_text(keys[idx])
        
    return fig


def apply_all_transformations(transformations, model, scene, init_camera_pos, original_img, rendered_img):
    # ---------------------------------------------------------
    # Transform and render image 
    # ---------------------------------------------------------

    # Rotate scene
    rotations = transformation_dict(azimuth = transformations['azimuth'], elevation = transformations['elevation'])
    roto_scene = transform_scene(model, scene,init_camera_pos, rotations)


    # Translate Scene
    translations = transformation_dict(x = transformations['xyz'][0], 
                                    y = transformations['xyz'][1], 
                                    z = transformations['xyz'][2])
    trans_scene = transform_scene(model, scene, init_camera_pos, translations)

    # Roto-Translate Scene
    roto_trans_scene = transform_scene(model, scene, init_camera_pos, transformations)



    # ---------------------------------------------------------
    # Loading and plotting the original image
    # ---------------------------------------------------------


    images = [original_img, rendered_img, roto_scene, trans_scene, roto_trans_scene]
    keys = ['Original Image', 'Rendered Image', 'Rotated Image', 'Translated Image', 'Roto-Translated Image']
    return images, keys
