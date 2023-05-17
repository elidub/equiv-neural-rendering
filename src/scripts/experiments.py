import json
import os
import sys
import time
import torch

src_path = os.path.join(sys.path[0], '../')
sys.path.insert(1, src_path)
from enr.misc.dataloaders import scene_render_dataloader
from enr.models.neural_renderer import NeuralRenderer, load_model
from enr.training.training import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get path to data from command line arguments
if len(sys.argv) != 2:
    raise(RuntimeError("Wrong arguments, use python experiments.py <config>"))
config_file = os.path.join(src_path, 'configs', sys.argv[1])

print('src_path: {}'.format(src_path))

# Open config file
with open(config_file) as file:
    config = json.load(file)

continue_training = True if config["timestamp"] is not False else False

# Set up directory to store experiments
timestamp = time.strftime("%Y-%m-%d_%H-%M") if not continue_training else config["timestamp"]
save_path = config["save_path"]
directory = "{}/{}_{}".format(os.path.join(src_path, save_path), timestamp, config["id"])
if not os.path.exists(directory):
    os.makedirs(directory)

print("Saving to {}".format(directory))

# Save config file in directory
with open(directory + '/config.json', 'w') as file:
    json.dump(config, file)

# Set up renderer
model = NeuralRenderer(
    img_shape=config["img_shape"],
    channels_2d=config["channels_2d"],
    strides_2d=config["strides_2d"],
    channels_3d=config["channels_3d"],
    strides_3d=config["strides_3d"],
    num_channels_inv_projection=config["num_channels_inv_projection"],
    num_channels_projection=config["num_channels_projection"],
    mode=config["mode"]
) if not continue_training else load_model(os.path.join(directory, "best_model.pt"))


if not continue_training:
    model.print_model_info()

model = model.to(device)

if config["multi_gpu"]:
    model = torch.nn.DataParallel(model)

# Set up trainer for renderer
trainer = Trainer(device, model, lr=config["lr"],
                  rendering_loss_type=config["loss_type"],
                  ssim_loss_weight=config["ssim_loss_weight"],
                  savedir = directory,)

dataloader = scene_render_dataloader(path_to_data=config["path_to_data"],
                                     batch_size=config["batch_size"],
                                     img_size=config["img_shape"],
                                     crop_size=config["crop_size"])

# Optionally set up test_dataloader
if config["path_to_test_data"]:
    test_dataloader = scene_render_dataloader(path_to_data=config["path_to_test_data"],
                                              batch_size=config["batch_size"],
                                              img_size=config["img_shape"],
                                              crop_size=config["crop_size"])
else:
    test_dataloader = None

print("PID: {}".format(os.getpid()))

# Check how many folders there are that have 'directory' as prefix	


# Train renderer, save generated images, losses and model
trainer.train(dataloader, config["epochs"], save_dir=directory,
              save_freq=config["save_freq"], test_dataloader=test_dataloader)

# Print best losses
print("Model id: {}".format(config["id"]))
print("Best train loss: {:.4f}".format(min(trainer.epoch_loss_history["total"])))
print("Best validation loss: {:.4f}".format(min(trainer.val_loss_history["total"])))
