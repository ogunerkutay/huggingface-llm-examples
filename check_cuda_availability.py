# FILE: check_cuda_availability.py
'''
This script checks the availability of CUDA (GPU support) using PyTorch.
It prints the PyTorch version, CUDA availability, CUDA version, number of GPUs, GPU details, and the file path of the installed PyTorch package.
Additionally, it runs the `nvidia-smi` command to fetch and print the NVIDIA System Management Interface output.
'''

import torch
import os

def test_pytorch_and_cuda():
    print("="*50)
    print("PyTorch & CUDA Environment Test")
    print("="*50)
    
    # PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    
    # CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"PyTorch is installed at: {torch.__file__}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  - Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"  - Memory Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
            print(f"  - Compute Capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("CUDA is NOT available!")
    print("="*50)

def check_nvidia_smi():
    print("\nNVIDIA-SMI Test")
    print("="*50)
    try:
        result = os.popen("nvidia-smi").read()
        print(result)
    except Exception as e:
        print(f"Error: Unable to fetch NVIDIA-SMI output. {e}")
    print("="*50)

def main():
    test_pytorch_and_cuda()
    check_nvidia_smi()

if __name__ == "__main__":
    main()
