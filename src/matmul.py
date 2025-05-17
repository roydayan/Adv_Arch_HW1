import torch
import matmul_cuda

def matmul(a1_tensor, a2_tensor):

    out_tensor = matmul_cuda.matmul_forward(a1_tensor, a2_tensor)

    return out_tensor
