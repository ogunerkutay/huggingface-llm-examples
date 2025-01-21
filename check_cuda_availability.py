# This script checks the availability of CUDA (GPU support) using PyTorch.
# The first line prints whether CUDA is available on the system (True if GPU is accessible, False otherwise).
# The second line prints the version of CUDA being used by PyTorch.
# The third line prints the name of the GPU device (if available) being used by PyTorch.
# The last line prints the file path of the installed PyTorch package to confirm the environment setup.
# This is useful for ensuring that the correct PyTorch installation and GPU setup are in place.

import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("Device Name:", torch.cuda.get_device_name(0))
print(torch.__file__) 