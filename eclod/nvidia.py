import torch
import os

# Set CUDA_LAUNCH_BLOCKING for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Clear GPU memory
torch.cuda.empty_cache()

# Check GPU memory usage
print("Before loading model:")
os.system("nvidia-smi")

# Load model on CPU first
ckpt_path = cfg.schedule.load_model if task > 2 else 'baseModels/CUMULtask1/model_last.ckpt'
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

# Assuming you have a model instance
model = build_model(cfg.model)

# Load state dict
model.load_state_dict(ckpt['state_dict'])

# Move model to GPU
model.to('cuda:0')

# Check GPU memory usage after loading model
print("After loading model:")
os.system("nvidia-smi")

# Continue with the rest of your script...