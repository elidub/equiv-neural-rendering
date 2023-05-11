import os

data_dir_train = '/project/gpuuva022/shared/equiv-neural-rendering/chairs/data_half/rot_dataset/train'
data_dir_valid = '/project/gpuuva022/shared/equiv-neural-rendering/chairs/data_half/rot_dataset/val'
data_dir_rest =  '/project/gpuuva022/shared/equiv-neural-rendering/chairs/data_half/rot_dataset/rest'

files = len(os.listdir(data_dir_train))
print('Train files:')
print(files)

files2 = len(os.listdir(data_dir_valid))
print('Valid files:')
print(files2)

files3 = len(os.listdir(data_dir_rest))
print('Rest files:')
print(files3)

total = files + files2 + files3
print('Total files:')
print(total)
# i = 0
# files = os.listdir(data_dir_train)
# for scene in os.listdir(data_dir_train) :
#     i += 1
#     origin = os.path.join(data_dir_train, scene)
#     if i > 2306:
#         os.system(f'rm -rf {origin}')
    # elif i < 2638:
    #     to = os.path.join(data_dir_valid, scene) + '/'
    #     print(f'cp -r  {origin} {to}')
    #     os.system(f'cp -r  {origin} {to}')
    # else:
    #     to = os.path.join(data_dir_rest, scene) + '/'
    #     print(f'cp -r  {origin} {to}')
    #     os.system(f'cp -r  {origin} {to}')
    

    
   