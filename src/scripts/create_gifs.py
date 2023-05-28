import matplotlib.pyplot as plt
import imageio
import torch
import torchvision
import sys, os
import argparse

# local_path = !pwd
# local_path = local_path[0]
# main_path = local_path[:-5]
# sys.path.append(local_path + '/../src/')
# sys.path.append(local_path + '/../src/enr/')
src_path = os.path.join(sys.path[0], '../')
sys.path.insert(1, src_path)

from torchvision.transforms import ToTensor
from enr.misc.viz import generate_novel_views
from enr.misc.utils import full_rotation_angle_sequence, sine_squared_angle_sequence, back_and_forth_angle_sequence, constant_angle_sequence
from enr.misc.viz import batch_generate_novel_views, save_img_sequence_as_gif
from enr.models.neural_renderer import load_model
from enr.transforms3d.conversions import deg2rad

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_img_tensor(img, nrow=4):
    """Helper function to plot image tensors.
    
    Args:
        img (torch.Tensor): Image or batch of images of shape 
            (batch_size, channels, height, width).
    """
    img_grid = torchvision.utils.make_grid(img, nrow=nrow)
    plt.imshow(img_grid.cpu().numpy().transpose(1, 2, 0))


def rotation_shifts(num_frames):

    num_frames = num_frames*3

    azimuth_shifts = full_rotation_angle_sequence(num_frames).to(device)
    # azimuth_shifts = constant_angle_sequence(num_frames).to(device) 
    elevation_shifts = sine_squared_angle_sequence(num_frames, -10., 20.).to(device)
    # elevation_shifts = constant_angle_sequence(num_frames).to(device) 
    translations_shifts = torch.zeros((num_frames, 3)).to(device)

    azimuth_shifts = deg2rad(azimuth_shifts)
    elevation_shifts = deg2rad(elevation_shifts)

    return azimuth_shifts, elevation_shifts, translations_shifts

def translation_shifts(num_frames):

    back_and_forth = back_and_forth_angle_sequence(num_frames, 0., .5)
    zeros = torch.zeros(len(back_and_forth))

    translations_shifts = torch.stack((
        torch.cat((back_and_forth, zeros, zeros), dim = 0),
        torch.cat((zeros, back_and_forth, zeros), dim = 0),
        torch.cat((zeros, zeros, back_and_forth), dim = 0),
    ), dim = 1).to(device)

    azimuth_shifts   = torch.zeros(len(translations_shifts)).to(device)
    elevation_shifts = torch.zeros(len(translations_shifts)).to(device)

    return azimuth_shifts, elevation_shifts, translations_shifts

def rototranslation_shifts(num_frames):

    azimuth_shifts = full_rotation_angle_sequence(num_frames*3).to(device)
    elevation_shifts = sine_squared_angle_sequence(num_frames*3, -10., 20.).to(device)

    back_and_forth = back_and_forth_angle_sequence(num_frames, 0., .5)
    zeros = torch.zeros(len(back_and_forth))

    translations_shifts = torch.stack((
        torch.cat((back_and_forth, zeros, zeros), dim = 0),
        torch.cat((zeros, back_and_forth, zeros), dim = 0),
        torch.cat((zeros, zeros, back_and_forth), dim = 0),
    ), dim = 1).to(device)

    azimuth_shifts = deg2rad(azimuth_shifts)
    elevation_shifts = deg2rad(elevation_shifts)

    return azimuth_shifts, elevation_shifts, translations_shifts

def get_shift(shift, num_frames):
    if shift == 'rot':
        azimuth_shifts, elevation_shifts, translations_shifts = rotation_shifts(num_frames)
    elif shift == 'trans':
        azimuth_shifts, elevation_shifts, translations_shifts = translation_shifts(num_frames)
    elif shift == 'rototrans':
        azimuth_shifts, elevation_shifts, translations_shifts = rototranslation_shifts(num_frames)
    else:
        print('shift not found')
    
    return azimuth_shifts, elevation_shifts, translations_shifts

def main(args):

    img_path = args.img_path
    model_path = args.model_path
    main_path = os.path.join(sys.path[0], '../')

    # Load trained chairs model
    model = load_model(main_path + model_path).to(device)
    img = imageio.imread(main_path + img_path)[:, :, :3]

    # Convert image to tensor and add batch dimension
    img_source = ToTensor()(img)
    img_source = img_source.unsqueeze(0).to(device)

    azimuth_source = torch.Tensor([0]).to(device)
    elevation_source = torch.Tensor([0]).to(device)
    translations_source = torch.Tensor([0., 0., 0.]).to(device)

    shifts = ['rot', 'trans', 'rototrans'] if args.shift == 'all' else [args.shift]

    views_all = []

    for shift in shifts:
        print(shift)
        angle_shift = get_shift(shift, args.num_frames)
        print([a.shape for a in angle_shift])
        views = batch_generate_novel_views(model, img_source, 
                                        azimuth_source, elevation_source, translations_source,
                                        *angle_shift)
        
        views_all.append(views)
        save_img_sequence_as_gif(views, main_path + 'imgs/gifs/' + args.save_name + f'_{shift}.gif')

    # # views_all is a list of lists of tensors, concatenate them
    # views_all = [torch.cat(views, dim=0) for views in views_all]
    # views_all = torch.cat(views_all, dim=0)

    # print(views_all.shape)


    # save_img_sequence_as_gif(views_all, main_path + 'imgs/gifs/' + args.save_name + f'_all.gif', nrow = 3)





def parse_option():

    parser = argparse.ArgumentParser(description='Create gifs')

    parser.add_argument('--img_path', type=str, default='imgs/example-data/00000.png')
    parser.add_argument('--model_path', type=str, default='train_results/2023-05-10_12-31_roto_lr2e-4/best_model.pt')
    parser.add_argument('--shift', type=str, default='rot', choices=['rot', 'trans', 'rototrans', 'all'])
    parser.add_argument('--save_name', type=str, default='rot')
    parser.add_argument('--num_frames', type=int, default=25)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_option()
    print(args)
    main(args)