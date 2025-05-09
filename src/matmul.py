import torch
import matmul_cuda

def matmul(a1_tensor, a2_tensor):
    #assert a1_tensor.shape == a2_tensor.shape
    #assert a1_tensor.dim() == 3 and a1_tensor.size(1) == 32 and a1_tensor.size(2) == 32
    #assert a1_tensor.device.type == 'cpu' and a2_tensor.device.type == 'cpu'
    #assert a1_tensor.dtype == torch.float32
    """
    if not a1_tensor.is_cuda:
        a1_tensor = a1_tensor.cuda()
    if not a2_tensor.is_cuda:
        a2_tensor = a2_tensor.cuda()
    """

    out_tensor = matmul_cuda.matmul_forward(a1_tensor, a2_tensor)

    #return out_tensor.cpu()
    return out_tensor