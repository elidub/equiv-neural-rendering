import os, argparse, sys, json
import random
import numpy as np
import math

def parse_option():

    parser = argparse.ArgumentParser(description='Create data for')
 
    parser.add_argument('--data_dir', type=str, default='/project/gpuuva022/shared/equiv-neural-rendering/chairs/data_one/rototrans_dataset')
    parser.add_argument('--n_scenes', type=int, default=1)
    parser.add_argument('--n_patches', type=int, default=3)

    args = parser.parse_args()
    return args


def in_patch(azi_angle, azi_patch, ele_angle, ele_patch, epsilon = 0.05):
    in_patch = True
    if azi_angle >= azi_patch + epsilon:
        in_patch = False
    if azi_angle <= azi_patch - epsilon:
        in_patch = False
    if ele_angle >= ele_patch + epsilon:
        in_patch = False
    if ele_angle <= ele_patch - epsilon:
        in_patch = False
    return in_patch

def main(args):
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')
    files = os.listdir(train_dir)

    for i in range(args.n_scenes):
    
        train_file = os.path.join(train_dir, files[i])
        val_file = os.path.join(val_dir, files[i])
        test_file = os.path.join(test_dir, files[i])
        test_params = {}
        new_render_param_train = {}
        new_render_param_val = {}

        removed = 0
        for j in range(args.n_patches): 
        # create random patch creation (+- epsilon)
            ele_patch = np.random.uniform(-math.pi/2, math.pi/2) # elevation
            azi_patch = np.random.uniform(-math.pi, math.pi) # azimuth
            name = "patch_{}".format(j)
            test_params[name] = {'azimuth': azi_patch, 'elevation': ele_patch, 'epsilon': 0.05}

            # Remove angles from json and 
            with open(os.path.join(train_file, 'render_params.json'), 'r') as f:
                train_params = json.load(f)
            
            # print(train_params)
            for key, value in train_params.items():
                # check if in patch
                # If so move this picture to test
                azi_angle = value['azimuth']
                ele_angle = value['elevation']
                if in_patch(azi_angle, azi_patch, ele_angle, ele_patch):
                    origin = os.path.join(train_file, key + '.png')
                    to = os.path.join(test_file)
                    os.system(f'cp {origin} {to}')
                    os.system(f'rm {origin}')
                    test_params[key] = value
                    removed += 1
                else:
                    new_render_param_train[key] = value
                # and remove them to test

            # Remove angles from json and 
            with open(os.path.join(val_file, 'render_params.json'), 'r') as f:
                val_params = json.load(f)
            
            for key, value in val_params.items():
                # check if in patch
                # If so remove this key and value
                azi_angle = value['azimuth']
                ele_angle = value['elevation']
                if in_patch(azi_angle, azi_patch, ele_angle, ele_patch):
                    origin = os.path.join(val_file, key + '.png')
                    to = os.path.join(test_file, key + '.png')
                    os.system(f'cp {origin} {to}')
                    os.system(f'rm {origin}')
                    test_params[key] = value
                    removed += 1
                else:
                    new_render_param_val[key] = value

            
        with open(test_file +"/render_params.json", "w") as f:
            json.dump(test_params, f)
        
        with open(train_file +"/render_params_new.json", "w") as f:
            json.dump(new_render_param_train, f)
        
        with open(val_file +"/render_params_new.json", "w") as f:
            json.dump(new_render_param_val, f)



if __name__ == '__main__':
    args = parse_option()
    print(args)
    main(args)
