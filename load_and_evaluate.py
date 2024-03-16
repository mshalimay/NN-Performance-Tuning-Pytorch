import argparse
import sys
import torch
from torchvision import datasets, transforms
from models.simplenet import SimpleNet
from models.resnet import ResNet, BasicBlock
from train import resmobile_options
import utils.utils as u


# set random seed
torch.manual_seed(1)

# baseline models checkpoints
model_paths = {
    'simplenet': './saved_models/simplenet_mnist_batch=64_epoch=14_lr=1.0_opt=adadelta_sched=step_nest=True.pt',
    'resnet': "./saved_models/resmobile_0_cifar10_batch=128_epoch=100_lr=0.1_opt=sgd_sched=cosine_nest=True.pt",
    'resmobile': "./saved_models/resmobile_cifar10_batch=128_epoch=100_lr=0.1_opt=sgd_sched=cosine_nest=True.pt"
}

def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description='Train resnet, resmobile or simplenet on mnist or cifar10')

    parser.add_argument('--model', type=str, default=None, choices=['simplenet', 'resnet', 'resmobile'],
                        help='neural network to train (default: simplenet)')

    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                        help='dataset to train on (default: mnist)')
    
    parser.add_argument('--model-path', type=str, default=None, help='path to model checkpoint')

    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='disables CUDA training (default: False)')

    parser.add_argument('--use-mps', action='store_true', default=False,
                        help='disables macOS GPU training')

    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for evaluation (default: 128)')

    return parser.parse_args()

def load_model(model_path:str):
    checkpoint = torch.load(model_path)
    # retrieve parameters from checkpoint
    in_shape = (checkpoint['in_shape'][0], checkpoint['in_shape'][1], checkpoint['in_shape'][2])
    dataset = checkpoint['args'].dataset.lower()
    num_classes = u.dataset_num_classes(dataset)

    if checkpoint['args'].model == 'simplenet':
        net = SimpleNet()
    elif checkpoint['args'].model == 'resnet':
        net = ResNet(img_channels=in_shape[0], num_layers=18, block=BasicBlock, num_classes=num_classes)
    elif checkpoint['args'].model == 'resmobile':
        # if there is ro in args, retrieve the corresponding mnet_conv
        if hasattr(checkpoint['args'], 'ro'):
            mnet_conv = resmobile_options[checkpoint['args'].ro]
        else:
            mnet_conv = [True, True, True, True]
        net = ResNet(img_channels=in_shape[0], num_layers=18, block=BasicBlock, num_classes=num_classes, mnet_conv=mnet_conv)
    else:
        raise NotImplementedError
    net.load_state_dict(checkpoint['model_state_dict'])
    return net

def load_prepare_data(dataset:str):
    if dataset.lower() == 'mnist':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        data_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
        data_test = datasets.MNIST('./data', train=False, transform=transform)
    elif dataset.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        data_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        data_test = datasets.CIFAR10('./data', train=False, transform=transform_test)
    else:
        raise NotImplementedError

    return data_train, data_test

def evaluate_model(net, device, test_loader):
    model = net.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model.forward(images)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / len(test_loader.dataset)
    return test_loss, accuracy

def main():
    args = parse_arguments()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    use_mps = args.use_mps and torch.backends.mps.is_available()

    if not args.model_path and not args.model:
        print("Please enter a model_path or model. Exiting...")
        sys.exit(0)

    model_path = args.model_path
    if not model_path:
        model_path = model_paths[args.model]

    # GPU/CPU settings
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load pytorch checkpoint from "./saved_models"
    checkpoint = torch.load(model_path)

    net = load_model(model_path)
    data_test = load_prepare_data(checkpoint['args'].dataset)[1]
  
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False)
    test_loss, accuracy = evaluate_model(net, device, test_loader)

    print(f"Test average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}%")

if __name__ == '__main__':
    main()