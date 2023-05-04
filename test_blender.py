
import matplotlib.pyplot as plt
import imageio
import torch
import torchvision
from skimage.transform import resize
import json
from transforms3d.conversions import rad2deg
from torchvision.transforms import ToTensor
from models.neural_renderer import load_model
import pickle
import argparse
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)

def img_tensor(img, nrow=4):
    """Helper function to plot image tensors.
    
    Args:
        img (torch.Tensor): Image or batch of images of shape 
            (batch_size, channels, height, width).
    """
    img_grid = torchvision.utils.make_grid(img, nrow=nrow)
    return img_grid.cpu().numpy().transpose(1, 2, 0)

def read_coords(img_id, data):
    azimuth_source = rad2deg(torch.Tensor([data[img_id]['azimuth']])).to(DEVICE)
    elevation_source = rad2deg(torch.Tensor([data[img_id]['elevation']])).to(DEVICE)
    translations_source = torch.Tensor([data[img_id]['x'], data[img_id]['y'], data[img_id]['z']]).to(DEVICE)
    return azimuth_source, elevation_source, translations_source

def render_source(img_source, model):
    img_source = ToTensor()(img_source)
    img_source = img_source.unsqueeze(0).to(DEVICE)

    # Infer scene representation
    scene = model.inverse_render(img_source)
    rendered = model.render(scene).detach()
    return img_tensor(rendered), scene

def render_target(
        scene, model,
        azimuth_source, elevation_source, translations_source,
        azimuth_target, elevation_target, translations_target
    ):
    
    rotated_scene = model.rotate_source_to_target(
        scene, 
        azimuth_source, elevation_source, translations_source,
        azimuth_target, elevation_target, translations_target
    )

    rendered = model.render(rotated_scene).detach()

    return img_tensor(rendered)

def plot_4_imgs(imgs, titles = ['source', 'target', 'rendered source', 'rendered target']):
    fig, axs = plt.subplots(1, 4, figsize=(8, 8))
    axs = axs.flatten()
    for i, img in enumerate(imgs):
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(titles[i])
    plt.show()


def produce_imgs(path, model, n = 10):
    with open(f'{path}/render_params.json') as f:
        data = json.load(f)

    imgs_all = []
    for i in tqdm(range(0, n, 2)):
        source_id, target_id = i, i + 1

        source_id, target_id = f'{source_id:05d}', f'{target_id:05d}'
        img_source = imageio.imread(f'{path}/{source_id}.png')[:, :, :3]
        img_target = imageio.imread(f'{path}/{target_id}.png')[:, :, :3]

        azimuth_source, elevation_source, translations_source = read_coords(source_id, data = data)
        azimuth_target, elevation_target, translations_target = read_coords(target_id, data = data)

        img_source_rendered, scene = render_source(img_source, model)
        img_target_rendered = render_target(
            scene, model,
            azimuth_source, elevation_source, translations_source,
            azimuth_target, elevation_target, translations_target
        )

        imgs = [img_source, img_source_rendered, img_target, img_target_rendered]
        imgs_all.append(imgs)

    return imgs_all

def plot_per_4_imgs(imgs_all, titles = ['source', 'rendered source', 'target', 'rendered target']):
    fig, axs_all = plt.subplots(len(imgs_all), 4, figsize = (10, 2*len(imgs_all)), tight_layout = True)
    for axs, imgs in zip(axs_all, imgs_all):
        for i, img in enumerate(imgs):
            axs[i].imshow(img)
            axs[i].axis('off')
            if (axs == axs_all[0]).all():
                axs[i].set_title(titles[i])
    plt.show()


############################################################################################################

def parse_option():
    parser = argparse.ArgumentParser(description="Plot renderings")
    parser.add_argument('--type', type=str, default='rot_dataset', help='type')
    args = parser.parse_args()
    return args


def main(args):
    # Load trained chairs model
    model = load_model('trained-models/chairs.pt').to(DEVICE)
    imgs_all = produce_imgs(path = f'data_prep/output/{args.type}/model.dae/', model = model, n = 50)

    # save imgs_all
    with open(f'figs/{args.type}.pkl', 'wb') as f:
        pickle.dump(imgs_all, f)


if __name__ == '__main__':
    args = parse_option()
    print(args)
    main(args)


"""
Run on GPU (really fast), or CPU (few minutes, depends on `n`)
python test_blender.py --type=rot_dataset
python test_blender.py --type=rototrans_dataset
python test_blender.py --type=trans_dataset
"""