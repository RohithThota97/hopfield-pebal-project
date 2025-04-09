import torch
print("PyTorch version:", torch.__version__)
print("CUDA version from PyTorch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Number of CUDA devices:", torch.cuda.device_count())