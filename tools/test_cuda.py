import torch
import sys

def test_cuda():
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("\nCUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        
        # Test basic GPU operation
        x = torch.rand(5, 3).cuda()
        print("\nSuccessfully created tensor on GPU:", x)
        print("Tensor device:", x.device)
        
        # Test simple computation
        y = x @ x.t()
        print("\nSuccessfully performed matrix multiplication on GPU")
        print("Result shape:", y.shape)
        print("Result device:", y.device)
    else:
        print("\nCUDA is not available. Please check your PyTorch installation and GPU drivers.")

if __name__ == "__main__":
    test_cuda() 