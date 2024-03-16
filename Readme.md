# Hyperparameter tuning and performance evaluation with Pytorch
This Repository:
- Implements the `ResNet18` model using Pytorch, with optional deptwhise-separable convolutions, as in MobileNet.
- Experiment with different hyperparameter settings, including: 
  - ResNet18 architechtures, schedulers, optimizers, learning rates, momentum, weight decay, etc. See `Usage` for a list of options.
- Evaluates performance of estimated models, including inference time, # params and MACs

See `Examples.md` for a simple run, hyperparameter exploration and performance evaluation

# Files and directories description
*   `./train.py`: script to train all the models with specified parameters
*   `./experiments.py`: script to run train the models with different parameters
*   `./performance.py`: script to compute performance metrics for a trained model
*   `./utils/utils.py`: contains functions common to all models
*   `./slurm_gpu.sbatch`: runs `experiments.py` in SLURM
*   `./models/`: directory containing the neural network classes
    *   `simplenet.py`: contains the neural network from `pytorch`'s [example](https://github.com/pytorch/examples/tree/main/mnist)
    *   `resnet.py`: contains the `Resnet-18` neural network, including the version with depthwise-separable convolutions
*   `./saved_models/`: directory containing checkpoints for trained neural networks
*   `./training_log/`: directory containing information collected during training

# Models short description
In this work there are three variations for the neural network architectures: _SimpleNet_, _ResNet_, _ResMobile_
## SimpleNet
*   _SimpleNet_ follows the neural network implemented in this [PyTorch's example](https://github.com/pytorch/examples/tree/main/mnist)
*   In short, is a neural network with two convolutional layers followed by two fully connected layers.
*   Please see `./models/simplenet.py` for complete details
## ResNet
*   _ResNet_ structure follows exactly the [original paper implementation](https://arxiv.org/pdf/1512.03385.pdf) (see figures 3 and table 1 for visuals) for the 18-layer case.
    
*   For what follows, and to understand the modifications in _ResMobile_, it is worth highlighting three features of the original architechture:
    
    *   Initial layer: this is a convolution applied directly to the raw inputs. It is indicated in the table below extracted from the original paper, with label _conv1_
    *   BasicBlocks: these are groups of convolutions stacked together in each of the _ResNet_ outer layers. For instance, in the table below, the _conv2\_x_ outer layer contains two BasicBlocks each of which contains two 3x3 x 64 convolutions.
    *   DownSampling layers: these are 1x1 convolutions used to match dimensions between outputs of convolutional layers and the shortcut connections.
*   Please see the original paper and `./models/resnet.py` for complete details

**Table: Architechtures for ResNet** 

![Alt text](<imgs/resnetArch.png>)

## ResMobile
*   _ResMobile_ has the exact same structure as _ResNet_, except that all convolutional layers in _ResNet_ are subject to change to a depthwise separable convolutional (DSC) layer.
    
*   The implementation of the depthwise separable convolution follows exactly the implementation in the [MobileNet paper](https://arxiv.org/abs/1704.04861)
    
    *   Particularly, a _ResNet_ convolution with input size `m`, output size `n` and `f` filters is substituted for
        *   Depthwise convolution: convolutional layer with input and output size `n` and same padding, kernel size, and stride as the respective convolution in _ResNet_, but where the filters are applied just to one channel of the input.
        *   A batch normalization layer
        *   A final 1x1 convolution, with input size `m`, output size `n` with stride=1, padding=0 and no bias applied to the depthwise convolution that outputs `n`
    *   The rest of the structure is the same as _ResNet_
*   For this project, I experimented with 5 options for the _ResMobile_, which vary by the place where the convolutional layers are substituted by DSC layers.
    
    *   Option 0: substitute all convolutional layers in _ResNet_ by DSC layers
    *   Option 1: substitute all convolutional layers in _ResNet_ by DSC layers, _except for the downsampling layers_
    *   Option 2: substitute all convolutional layers in _ResNet_ by DSC layers, _except for the initial ResNet layer_
    *   Option 3: substitute all convolutional layers in _ResNet_ by DSC layers, _except for the initial ResNet layer_ and the _downsampling layers_
    *   Option 4: _only the first convolutional layer of a basic block_ is substituted by a DSC layer
    *   Option 5: _only the second convolutional layer of a basic block_ is substituted by a DSC layer
*   See `./models/resnet.py` for complete details
    
# Usage
## Model training
To train any model with default parameters, navigate to the **root folder** and run:

*   `python training.py <model_name>`
*   where `model_name`:
    *   `simplenet`: for the neural network in Pytorch's example
    *   `resnet`: for ResNet18
    *   `resmobile`: for ResNet18 with depthwise-separable convolutions

**Additional options**  
The `training.py` script have a couple of options to tweak the training scheme. To see the full list, run `training.py -h`, which outputs the below list:

```bash
train.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N] [--lr LR] [--gamma M] [--no-cuda] [--no-mps] [--dry-run] [--seed S] [--log-interval N] [--save-model] [--model {simplenet,resnet,resmobile}] [--dataset {mnist,cifar10}] [--log-train] [--o {adadelta,sgd,adam,adamw}] [--sched {step,plateau,cosine,cosine_r,cyclic,none}] [--w-decay W_DECAY] [--momentum MOMENTUM] [--nest NEST] [--ro RO]
    
    Train resnet, resmobile or simplenet on mnist or cifar10
    
    options:
      -h, --help            show this help message and exit
      --batch-size N        input batch size for training (default: 64)
      --test-batch-size N   input batch size for testing (default: 1000)
      --epochs N            number of epochs to train (default: 14)
      --lr LR               learning rate (default: 1.0)
      --gamma M             Learning rate step gamma (default: 0.7)
      --no-cuda             disables CUDA training
      --no-mps              disables macOS GPU training
      --dry-run             quickly check a single pass
      --seed S              random seed (default: 1)
      --log-interval N      print training status every `log-interval` batches. Use -1 to disable (default: 20)
      --save-model          For Saving the best Model
      --model {simplenet,resnet,resmobile}
                            neural network to train (default: simplenet)
      --dataset {mnist,cifar10}
                            dataset to train on (default: mnist)
    ne)
      --w-decay W_DECAY     weight decay for optimizer (default: 5e-4)
      --momentum MOMENTUM   momentum for SGD optimizer (default: 0.9)
      --nest NEST           Use Nesterov momentum in SGD optimizer (default: True)
      --ro RO               ResMobile option (default: 0) 
      0: Apply depthwise convolution in all convolution layers. 
      1: No depthwise convolution ResNet downsampling layer. 
      2: No depthwise convolution in ResNet initial layer. 
      3: No depthwise convolution in ResNet initial layer and downsampling layer.   
      4: Depthwise convolution only in the 2nd convolution of a basic block. 
      5: Depthwise convolution only in the 1st convolution of a basic block.
    
```

## Model performance
The file `performance.py` generates performance metrics for a trained model and save the results to a xlsx file, including:
- MACs
- Inference Latency (mean and std)
- Number of parameters

**Usage:**
```shell
performance.py [-h] [--repetitions REPETITIONS] [--batchsize BATCHSIZE] [--search-dir SEARCH_DIR] [--model-path MODEL_PATH] [--seed SEED]

options:
  -h, --help            show this help message and exit
  --repetitions REPETITIONS
                        Number of repetitions to measure performance
  --batchsize BATCHSIZE
                        Batch size for latency measurement
  --search-dir SEARCH_DIR
                        Directory to search for models. If provided, all models in the directory will be evaluated.
  --model-path MODEL_PATH
                        Path to model checkpoint
  --seed SEED           Random seed
```

# References

The Correct Way to Measure Inference Time of Deep Neural Networks: https://deci.ai/blog/measure-inference-time-deep-neural-networks/

How to Accurately Time CUDA Kernels in Pytorch: https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch

Implementing ResNet18 in PyTorch from Scratch: https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/

Flops counter pytorch: https://github.com/sovrasov/flops-counter.pytorch

SGD with warm restart (cosine annealing with restart paper): https://arxiv.org/abs/1608.03983
 
