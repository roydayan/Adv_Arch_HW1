import torch
import matmul_cuda

def matmul(a1_tensor, a2_tensor):
    """
    Multiply batches of 32x32 matrices using a custom CUDA kernel.
    Args:
        a1_tensor: torch.Tensor of shape [N, 32, 32], CPU, float32
        a2_tensor: torch.Tensor of shape [N, 32, 32], CPU, float32
    Returns:
        out_tensor: torch.Tensor of shape [N, 32, 32], CPU, float32
    """
    assert a1_tensor.shape == a2_tensor.shape
    assert a1_tensor.shape[1:] == (32, 32)
    assert a1_tensor.device.type == 'cpu' and a2_tensor.device.type == 'cpu'
    # Move to CUDA
    a1_cuda = a1_tensor.cuda()
    a2_cuda = a2_tensor.cuda()
    # Call CUDA extension
    out_cuda = matmul_cuda.forward(a1_cuda, a2_cuda)
    # Move back to CPU
    return out_cuda.cpu()
