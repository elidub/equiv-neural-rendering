import os, argparse, sys
import random

def parse_option():

    parser = argparse.ArgumentParser(description='Create data for')

    parser.add_argument('--blender_dir', type=str, default='~/equiv-neural-rendering/src/blender/blender')
    parser.add_argument('--obj_dir', type=str, default='/project/gpuuva022/shared/equiv-neural-rendering/chairs/objects/')
    parser.add_argument('--output_dir', type=str, default='/project/gpuuva022/shared/equiv-neural-rendering/chairs/data_all/')
    parser.add_argument('--transformation', type=str, default='rototrans', choices = ['rototrans', 'rotation', 'translation'])

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
    parser.add_argument('--resolution', type=int, default=128, # 128
                        help='Resolution of the images.')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    return args

def main(args):
    files = os.listdir(args.obj_dir)
    random.shuffle(files)
    i = 0
    for i, scene in enumerate(files):
        i += 1
        scene_folder = os.path.join(args.obj_dir, scene)

        transformation_dict = {
        #   'transformation' : [transf_output_dir,   'arguments for blender script'],
            'rototrans'      : ['rototrans_dataset', '--rotation --translation'], 
            'rotation'       : ['rot_dataset',        '--rotation'],
            'translation'    : ['trans_dataset',     '--translation']
        }

        transf_output_dir, blender_args = transformation_dict[args.transformation]

        output_dir = os.path.join(args.output_dir, transf_output_dir)

        if i < 4613:
            output_dir = os.path.join(output_dir, 'train')
        elif i < 5275:
            output_dir = os.path.join(output_dir, 'val')
        else:
            output_dir = os.path.join(output_dir, 'rest')

        render_blender_file = os.path.join(sys.path[0], '../enr/data/render_blender.py')

        os.system(f'{args.blender_dir} -b --python {render_blender_file} -- --scene_name {scene} --scene_folder {scene_folder} --n_images {args.n_images} --output_folder {output_dir} --resolution {args.resolution} {blender_args}')
        


if __name__ == '__main__':
    args = parse_option()
    print(args)
    main(args)