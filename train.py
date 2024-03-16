from __future__ import print_function
import argparse
import os
import time
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from models.simplenet import SimpleNet
from models.resnet import ResNet, BasicBlock
import utils.utils as u

resmobile_options = {
    0: [True, True, True, True],     # depthwise in all convolutions
    1: [True, False, True, True],    # NOT depthwise in downsampling layer
    2: [False, True, True, True],    # NOT depthwise in resnet initial layer
    3: [False, False, True, True],   # NOT depthwise in resnet initial layer and downsampling
    4: [False, False, False, True],  # depthwise only in 2nd convolution of basic block
    5: [False, False, True, False],  # depthwise only in 1st convolution of basic block
}

def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description='Train resnet, resmobile or simplenet on mnist or cifar10')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')

    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')

    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')

    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='print training status every `log-interval` batches. Use -1 to disable (default: 20)')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the best Model')


    parser.add_argument('--model', type=str, default='simplenet', choices=['simplenet', 'resnet', 'resmobile'],
                        help='neural network to train (default: simplenet)')

    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                        help='dataset to train on (default: mnist)')
    
    parser.add_argument('--log-train', action='store_true', default=False,
                        help='save training logs to csv (default: False)')

    parser.add_argument('--o', type=str, default='sgd', choices=['adadelta', 'sgd', 'adam', 'adamw'],
                        help='optimizer to use (default: sgd)')

    parser.add_argument('--sched', type=str, default='cosine', choices=['step', 'plateau', 'cosine', 'cosine_r', 'cyclic', 'none'],
                        help='learning rate scheduler to use (default: cosine)')
    
    parser.add_argument('--w-decay', type=float, default=5e-4,
                        help='weight decay for optimizer (default: 5e-4)')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD optimizer (default: 0.9)')

    parser.add_argument('--nest', type = int, default = 1,  choices=[0, 1],
                        help='Use Nesterov momentum in SGD optimizer (default: 1)')

    parser.add_argument('--ro', type=int, default=0,
                    help=('ResMobile option (default: 0)\n'
                          '0: Apply depthwise convolution in all convolution layers.\n'
                          '1: No depthwise convolution ResNet downsampling layer.\n'
                          '2: No depthwise convolution in ResNet initial layer.\n'
                          '3: No depthwise convolution in ResNet initial layer and downsampling layer.\n'
                          '4: Depthwise convolution only in the 2nd convolution of a basic block.\n'
                          '5: Depthwise convolution only in the 1st convolution of a basic block.'))

    return parser.parse_args()

def main():
    #===========================================================================
    # parse arguments and set up
    #===========================================================================
    args = parse_arguments()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # set random seed
    torch.manual_seed(args.seed)

    # CUDA settings
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    model_name = f"{args.model}_{args.ro}" if args.model.lower() == 'resmobile' else args.model

    # create directory to save models, if specified
    checkpoint_filename = None
    if args.save_model:
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        best_accuracy = -float('inf')
        checkpoint_filename = (f"./saved_models/{model_name}_{args.dataset}_batch={args.batch_size}"
                            f"_epoch={args.epochs}_lr={args.lr}_opt={args.o}_sched={args.sched}_nest={args.nest}.pt")

    # create training log file, if specified
    train_log_filename = None
    if args.log_train:
        if not os.path.exists('./training_log'): 
            os.makedirs('training_log')

        # overwrite existing file
        train_log_filename = (f"./training_log/{model_name}_{args.dataset}_batch={args.batch_size}"
                            f"_epoch={args.epochs}_lr={args.lr}_opt={args.o}_sched={args.sched}_nest={args.nest}.csv")
        
        with open(train_log_filename, 'w') as f:
            f.write("epoch,test_loss,accuracy,time_elapsed\n")
    #===========================================================================
    # load and prepare datasets
    #===========================================================================
    # TODO: modularize this part
    # load and preprocess data
    if args.dataset.lower() == 'mnist':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    elif args.dataset.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        dataset1 = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        dataset2 = datasets.CIFAR10('./data', train=False, transform=transform_test)
    else:
        raise NotImplementedError
    
    # create data loaders
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    #===========================================================================
    # instantiate and prepare model
    #===========================================================================
    # instantiate the desired model and transfer to device
    num_classes = u.dataset_num_classes(args.dataset)
    if args.model.lower() == 'simplenet':
        net = SimpleNet()
    elif args.model.lower() == 'resnet':
        net = ResNet(img_channels=dataset1[0][0].shape[0], num_layers=18, block=BasicBlock, num_classes=num_classes)
    elif args.model.lower() == 'resmobile':
        net = ResNet(img_channels=dataset1[0][0].shape[0], num_layers=18, block=BasicBlock, num_classes=num_classes, 
                     mnet_conv=resmobile_options[args.ro])
    else:
        raise NotImplementedError

    # define the loss function
    loss = torch.nn.CrossEntropyLoss()

    # warm up GPU before training
    if use_cuda:
        u.GPU_warmup(device, warmup_iterations=40)

    # transfer model to device
    model = net.to(device)

    # define the optimizer
    if args.o.lower() == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    elif args.o.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                              weight_decay=args.w_decay, dampening=0, nesterov=args.nest)
    elif args.o.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    else:
        raise NotImplementedError
    
    # define the learning rate scheduler
    use_metric = True if args.sched.lower() == 'plateau' else False
    if args.sched.lower() == 'step':
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    elif args.sched.lower() == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
    elif args.sched.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.sched.lower() == 'cosine_r':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0)
    elif args.sched.lower() == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 1e-5, max_lr = args.lr, 
                                                      step_size_up = 5, step_size_down=10, mode = "triangular")
    elif args.sched.lower() == 'none':
        scheduler = None
    else:
        raise NotImplementedError

    #===========================================================================
    # train-test loop
    #===========================================================================
    # Time in training is just being measured after each epoch, so can do with simple python timer.
    #   obs: perhaps not the most accurate, but good enough for now.
    t0, t1, time_elapsed = time.time(), time.time(), 0
    current_lr = args.lr  # keep track of current learning rate to print changes
    num_epochs = 1 if args.dry_run else args.epochs  # if dry run, only train for 1 epoch
    best_accuracy = -float('inf')  # keep track of best accuracy to save model

    train_model = u.train
    if args.log_interval < 0:
        train_model = u.train_noprint

    for epoch in range(num_epochs):
        # train and test the model
        train_model(args.log_interval, model, loss, device, train_loader, optimizer, epoch)
        test_loss, accuracy = u.test(model, device, loss, test_loader)

        if scheduler is not None:
            if use_metric:
                scheduler.step(test_loss)
            else:
                scheduler.step()

        # accumulate time elapsed
        t1 = time.time()
        time_elapsed += t1 - t0
        t0 = t1
        
        # update learning rate and print if changed        
        if optimizer.param_groups[0]['lr'] != current_lr:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate changed to {current_lr}")

        # Check if the current model is the best and save if specified
        if args.save_model and accuracy > best_accuracy:
            best_accuracy = accuracy

            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': test_loss, 
                        'accuracy': accuracy, 'in_shape':dataset1[0][0].shape,'args': args}, checkpoint_filename)

            print(f"Accuracy improved. Saved model to {checkpoint_filename}")

        # save training logs to csv if specified
        if args.log_train:
            u.log_training(train_log_filename, accuracy, test_loss, epoch, time_elapsed)

if __name__ == '__main__':
    main()
