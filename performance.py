"""
Generates performance metrics for a trained model, including:
- MACs
- Inference Latency (mean and std)
- # Parameters
Saves the results to xlsx file.
"""

import os
import numpy as np
import pandas as pd
import torch
from ptflops import get_model_complexity_info
import utils.utils as u
from models.simplenet import SimpleNet
from models.resnet import ResNet, BasicBlock
from train import resmobile_options
import argparse


def performance_metrics(checkpoint_path, device, repetitions=1000, batchsize = 1):
    # Load pytorch checkpoint from "./saved_models"
    checkpoint = torch.load(checkpoint_path)

    # retrieve parameters from checkpoint
    in_shape = (checkpoint['in_shape'][0], checkpoint['in_shape'][1], checkpoint['in_shape'][2])
    num_classes = u.dataset_num_classes(checkpoint['args'].dataset)

    if checkpoint['args'].model == 'simplenet':
        model_name = 'simplenet'
        net = SimpleNet()
    elif checkpoint['args'].model == 'resnet':
        model_name = 'resnet'
        net = ResNet(img_channels=in_shape[0], num_layers=18, block=BasicBlock, num_classes=num_classes)
    elif checkpoint['args'].model == 'resmobile':
        # if there is ro in args, retrieve the corresponding mnet_conv
        if hasattr(checkpoint['args'], 'ro'):
            model_name = f"resmobile_{checkpoint['args'].ro}"
            mnet_conv = resmobile_options[checkpoint['args'].ro]
        else:
            model_name = 'resmobile_0'
            mnet_conv = [True, True, True, True]
        net = ResNet(img_channels=in_shape[0], num_layers=18, block=BasicBlock, num_classes=num_classes, mnet_conv=mnet_conv)
    else:
        raise NotImplementedError

    print(f"Loading model: {model_name}")

    net.load_state_dict(checkpoint['model_state_dict']) 

    model = net.to(device)
    model.eval()

    # Init time loggers
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions,1))


    # Warm up GPU
    dummy_data = torch.randn(batchsize, in_shape[0], in_shape[1], in_shape[2], dtype=torch.float).to(device)
    for _ in range(50):
        _ = model.forward(dummy_data)

    # Measure performance
    with torch.no_grad():
        for rep in range(repetitions):
            dummy_data = torch.randn(1, in_shape[0], in_shape[1], in_shape[2], dtype=torch.float).to(device)
            starter.record()
            _ = model.forward(dummy_data)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)

    macs, params = get_model_complexity_info(model, in_shape, as_strings=True,
                                        print_per_layer_stat=False, verbose=True)

    return macs, params, mean_syn, std_syn, model_name

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Compute MMACs, # params, and inference latency metrics for a given model.')
    parser.add_argument('--repetitions', type=int, default=1000, help='Number of repetitions to measure performance')
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size for performance measurement')
    parser.add_argument('--search-dir', type=str, default=None, help='Directory to search for models. If provided, all models in the directory will be evaluated.')
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.seed)
        
    # CUDA settings
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # list checkpoint files
    ckpt_files = []
    if args.model_path:
        ckpt_files.append(args.model_path)
    if args.search_dir:
        for file in os.listdir(args.search_dir):
            if file.endswith(".pt"):
                ckpt_files.append(os.path.join(args.search_dir, file))

    # get performance metrics for each model
    perf_data = []
    for ckpt_file in ckpt_files:
        macs, params, mean_syn, std_syn, model = performance_metrics(ckpt_file, device, args.repetitions, args.batchsize)
        perf_data.append([model, macs, params, mean_syn, std_syn])

    # save results to excel file
    df = pd.DataFrame(perf_data, columns=['model', 'macs', 'params', 'mean_syn', 'std_syn'])
    df.to_excel('experiments_performance.xlsx')


if __name__ == "__main__":
    main()