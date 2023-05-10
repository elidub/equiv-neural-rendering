import os, argparse, sys
import random

parser = argparse.ArgumentParser(description='Create data for')
parser.add_argument('--n_images', type=int, default=50,
                    help='number of views/images to be rendered per scende')
parser.add_argument('--data_folder', type=str,
                    help='Path to the dataset with .dae objects')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.') # ?????
parser.add_argument('--depth_scale', type=float, default=1.4,
                    help='Scaling that is applied to depth. Depends on size of mesh.') # ????
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.') # ????
parser.add_argument('--resolution', type=int, default=64, # 128
                    help='Resolution of the images.')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

obj_dir = '/project/gpuuva022/shared/equiv-neural-rendering/objects/dataset/'
data_dir_train = '/project/gpuuva022/shared/equiv-neural-rendering/data/half_data/train'
data_dir_valid = '/project/gpuuva022/shared/equiv-neural-rendering/data/half_data/valid'

#output_datasets = ["output/rot_dataset/", "output/trans_dataset/", "output/rototrans_dataset/", "output/test/"]
#output_dummy_datasets = ["output_dummy/rot_dataset/", "output_dummy/trans_dataset/", "output_dummy/rototrans_dataset/"]
i = 0
files = os.listdir(obj_dir) 
random.shuffle(files)
for scene in files:
    scene_folder = obj_dir + scene
    
    if i < 2306:
        os.system(f'~/equiv-neural-rendering/blender/blender -b --python data_prep/render_blender.py -- --scene_name {scene} --scene_folder {scene_folder} --n_images {args.n_images} --output_folder {data_dir_train} --rotation')
    
    elif i < 2637:
        os.system(f'~/equiv-neural-rendering/blender/blender -b --python data_prep/render_blender.py -- --scene_name {scene} --scene_folder {scene_folder} --n_images {args.n_images} --output_folder {data_dir_valid} --rotation')
    
    i += 1
    # Do for translation only
    #os.system(f'~/equiv-neural-rendering/blender/blender -b --python render_blender.py -- --scene_name {scene_name} --n_images {args.n_images} --output_folder {output_datasets[1]} --translation') 
    
    # Do for rotation and translation only
    #os.system(f'~/equiv-neural-rendering/blender/blender -b --python render_blender.py -- --scene_name {scene_name} --n_images {args.n_images} --output_folder {output_datasets[2]}  --rotation --translation')