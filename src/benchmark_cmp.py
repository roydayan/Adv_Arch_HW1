from __future__ import division
from __future__ import print_function

import argparse
from cgitb import reset

import numpy as np

import torch
from sympy.codegen import Print
from sympy.codegen.ast import float32
from torch.cuda import device
from triton.language import dtype

from matmul import matmul
import torch.backends.cudnn as cudnn
import gc

def compute_run_args(args):
    if args.gpus is not None and not args.force_cpu and torch.cuda.is_available():
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.enabled = True
        cudnn.benchmark = True
        args.device = torch.device('cuda:' + str(args.gpus[0]))
        torch.cuda.manual_seed(args.seed)
    else:
        args.gpus = []
        args.device = torch.device('cpu')
    torch.cuda.set_device(args.device)
    torch.cuda.init()
    print('Running inference on device \"{}\"'.format(args.device))
    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: random)')
    parser.add_argument('--gpus', default='0', help='List of GPUs used - e.g 0,1,3')
    parser.add_argument('--force_cpu', action='store_true', help='Force pytorch to run in CPU mode.')
    parser.add_argument('--num_examples', type=int, default=1000)
    parser.add_argument('--num_runs', type=int, default=500)
    parser.add_argument('--data_n', type=int, default=32)
    parser.add_argument('--tscale', choices=['s', 'ms', 'us'], default='ms')
    parser.add_argument('--dtype', choices=['float32', 'bool'], default='float32')
    # parser.add_argument('-s', '--state-size', type=int, default=128)
    # parser.add_argument('-b', '--batch-size', type=int, default=16)
    # parser.add_argument('-d', '--double', action='store_true')
    args = parser.parse_args()
    
    args = compute_run_args(args)
    args.dtype_dict = {'float32': torch.float32, 'bool': torch.bool}
    args.dtype_str = args.dtype
    args.dtype = args.dtype_dict[args.dtype]
    args.data_shape = [args.data_n, args.data_n]
    args.examples_shape = [args.num_examples, args.data_n, args.data_n]
    args.tscale_dict = {'s': 1, 'ms': 1000, 'us': 1000000}
    args.tscale_str = args.tscale
    args.tscale = args.tscale_dict[args.tscale]
    args.kwargs = {'dtype': args.dtype,
              'device': args.device,
              'requires_grad': True}
    
    print("Parameter parsing finished")
    print("device: ", args.device)
    print("dtype_str: ", args.dtype_str)
    print("dtype: ", args.dtype)
    print("data_shape: ", args.data_shape)
    print("examples_shape: ", args.examples_shape)
    print("tscale_str: ", args.tscale_str)
    print("tscale: ", args.tscale)
    return args

def random_uniform(examples_shape, dtype, device):
    return torch.empty(examples_shape, dtype=dtype, device=device).uniform_(-1, 1)

def random_bool(examples_shape, dtype, device):
    return torch.empty(examples_shape, dtype=dtype, device=device).bernoulli_()

def gt_matmul_uniform(A1, A2):
    return A1 @ A2

def gt_matmul_bool(A1, A2):
    return (A1.to(dtype=torch.float32) @ A2.to(dtype=torch.float32)).to(dtype=torch.bool)
    
def diff_uniform(gt, result):
    return gt - result
    
def diff_bool(gt, result):
    return gt.logical_xor(result).to(torch.float32)

def run(args):
    with (torch.no_grad()):
        all_compute_time_mean = []
        all_mean_L1_error = []
        all_mean_L2_error = []
        
        print("Setting computational methods based on data type")
        if args.dtype_str == 'bool':
            random_init = random_bool
            gt_matmul = gt_matmul_bool
            diff_method = diff_bool
        else:
            random_init = random_uniform
            gt_matmul = gt_matmul_uniform
            diff_method = diff_uniform

        print("Setting List or Tensor input type:")
        # A1_tensor = torch.empty(args.examples_shape, dtype=args.dtype, device=args.device).uniform_(-1, 1)
        A1_tensor = random_init(args.examples_shape, args.dtype, args.device)
        # A2_tensor = torch.empty(args.examples_shape, dtype=args.dtype, device=args.device).uniform_(-1, 1)
        A2_tensor = random_init(args.examples_shape, args.dtype, args.device)
        input_istensor = True
        try:
            matmul(A1_tensor, A2_tensor)
        except:
            print("Input type set as a list of torch Tensors")
            input_istensor = False
        else:
            print("Input type set as a single torch Tensor")
        del A1_tensor
        del A2_tensor
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        for run_idx in range(args.num_runs):
            print("run number: ", run_idx)
            print("initializing random inputs and computing ground truth")
            A1_tensor = random_init(args.examples_shape, args.dtype, args.device)
            A2_tensor = random_init(args.examples_shape, args.dtype, args.device)
            gt_tensor = gt_matmul(A1_tensor, A2_tensor)
            A1_lst = A1_tensor.split(split_size=1, dim=0)
            A2_lst = A2_tensor.split(split_size=1, dim=0)
            print("A1_tensor.shape: ", A1_tensor.shape)
            print("A1_tensor.device: ", A1_tensor.device)
            print("A2_tensor.shape: ", A2_tensor.shape)
            print("A2_tensor.device: ", A2_tensor.device)
            print("gt_tensor.shape: ", gt_tensor.shape)
            print("gt_tensor.device: ", gt_tensor.device)
            print("len(A1_lst): ", len(A1_lst))
            print("len(A2_lst): ", len(A2_lst))
            if input_istensor:
                A1_input = A1_tensor
                A2_input = A2_tensor
            else:
                A1_input = A1_lst
                A2_input = A2_lst
            
            print("Running matmul cuda implementation")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            matmul_res = matmul(A1_input, A2_input)
            end_event.record()
            torch.cuda.synchronize()

            # --- BEGIN: Added for PyTorch timing comparison ---
            print("Running PyTorch built-in matmul for timing comparison")
            torch_start_event = torch.cuda.Event(enable_timing=True)
            torch_end_event = torch.cuda.Event(enable_timing=True)
            torch_start_event.record()
            torch_matmul_res = A1_tensor @ A2_tensor
            torch_end_event.record()
            torch.cuda.synchronize()
            # --- END: Added for PyTorch timing comparison ---

            print("Computation finished, processing results")
            compute_time_total = start_event.elapsed_time(end_event) * args.tscale
            compute_time_mean = compute_time_total / args.num_examples
            # --- BEGIN: Added for PyTorch timing comparison ---
            torch_compute_time_total = torch_start_event.elapsed_time(torch_end_event) * args.tscale
            torch_compute_time_mean = torch_compute_time_total / args.num_examples
            print("[Timing] Custom matmul total: ", compute_time_total, args.tscale_str, ", mean: ", compute_time_mean, args.tscale_str)
            print("[Timing] PyTorch matmul total: ", torch_compute_time_total, args.tscale_str, ", mean: ", torch_compute_time_mean, args.tscale_str)
            # --- END: Added for PyTorch timing comparison ---
            if not input_istensor:
                res_tensor = torch.stack(matmul_res, dim=0).to(args.device)
            else:
                res_tensor = matmul_res.to(args.device)
            res_isfinite = res_tensor.isfinite()
            res_notfinite_num = res_isfinite.logical_not().sum()
            res_isfinite_ratio = res_isfinite.sum() / res_isfinite.numel()
            print("compute_time_total: ", compute_time_total, " ", args.tscale_str)
            print("compute_time_mean: ", compute_time_mean, " ", args.tscale_str)
            print("res_tensor.shape: ", res_tensor.shape)
            print("res_notfinite_num: ", res_notfinite_num)
            print("res_isfinite_ratio: ", res_isfinite_ratio)
    
            diff = diff_method(gt_tensor, res_tensor).view(args.num_examples, -1)
            mean_L1_error = diff.norm(p=1,dim=1).mean(0).item()
            mean_L2_error = diff.norm(p=2,dim=1).mean(0).item()
            print("diff_L1: ", mean_L1_error)
            print("diff_L2: ", mean_L2_error)
            
            all_compute_time_mean.append(compute_time_mean)
            all_mean_L1_error.append(mean_L1_error)
            all_mean_L2_error.append(mean_L2_error)
            
            del A1_tensor
            del A2_tensor
            del gt_tensor
            del A1_lst
            del A2_lst
            del A1_input
            del A2_input
            del start_event
            del end_event
            del matmul_res
            del compute_time_total
            del compute_time_mean
            del res_tensor
            del res_isfinite
            del res_notfinite_num
            del res_isfinite_ratio
            del diff
            del mean_L1_error
            del mean_L2_error
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            
        
        print("Printing Summary over runs:")
        overall_compute_time_mean = np.mean(all_compute_time_mean)
        overall_mean_L1_error = np.mean(all_mean_L1_error)
        overall_mean_L2_error = np.mean(all_mean_L2_error)
        if input_istensor:
            print("Input type was set as a single torch Tensor")
        else:
            print("Input type was set as a list of torch Tensors")
        print("overall mean computation time: ", overall_compute_time_mean, " ", args.tscale_str)
        print("overall mean L1 error: ", overall_mean_L1_error)
        print("overall mean L2 error: ", overall_mean_L2_error)
        
if __name__ == '__main__':
    args = parse_args()
    run(args)