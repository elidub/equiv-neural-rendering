import sys
import torch
import os

src_path = os.path.join(sys.path[0], '../')
sys.path.insert(1, src_path)

from enr.misc.dataloaders import scene_render_dataset
from enr.misc.quantitative_evaluation import get_dataset_psnr
from enr.models.neural_renderer import load_model

def evaluate(model_path, data_dir):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Load model
	model = load_model(model_path)
	model = model.to(device)

	# Initialize dataset
	dataset = scene_render_dataset(path_to_data=data_dir, img_size=(3, 128, 128),
		                       crop_size=128, allow_odd_num_imgs=True)

	# Calculate PSNR
	with torch.no_grad():
	    psnrs = get_dataset_psnr(device, model, dataset, source_img_idx_shift=64,
		                     batch_size=1, max_num_scenes=None)
