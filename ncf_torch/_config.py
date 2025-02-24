#%%
import torch

print("\n############# Neural Context Flow #############\n")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name())
    print("CUDA device capability:", torch.cuda.get_device_capability())
    print("CUDA device memory:", torch.cuda.get_device_properties(0).total_memory)

if __name__ == 'main':
    print("\n###############################################\n")
