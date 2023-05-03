import os, argparse, sys

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
parser.add_argument('--resolution', type=int, default=128, # 600
                    help='Resolution of the images.')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

output_datasets = ["output/rot_dataset/", "output/trans_dataset/", "output/rototrans_dataset/"]


for scene in os.listdir("dataset"): 
    scene_name = "dataset/" + scene
    # Do for rotation only
    os.system(f'blender --background --python render_blender.py -- --scene_name {scene_name} --n_images {args.n_images} --rotation')
    
    # Do for translation only
    os.system(f'blender --background --python render_blender.py -- --scene_name {scene_name} --n_images {args.n_images} --output_folder {output_datasets[1]} --translation')
    
    # Do for rotation and translation only
    os.system(f'blender --background --python render_blender.py -- --scene_name {scene_name} --n_images {args.n_images} --output_folder {output_datasets[2]}  --rotation --translation')