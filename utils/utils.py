from __future__ import print_function
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# functions to train and test neural net models

# Training
def train(log_interval, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()        
        if log_interval > 0 and batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Training
def train_noprint(log_interval, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()        



def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.4f}%)\n')

    return test_loss, accuracy


def log_training(log_path:str, accuracy:float, test_loss:float, epoch:float, time_elapsed:float) -> None:
    # if no CSV in log_path, create one
    with open(log_path, 'a') as f:
        # save results to csv
        f.write(f"{epoch},{test_loss},{accuracy},{time_elapsed}\n")
        

def dataset_num_classes(dataset_name:str):
    num_classes = 0
    if dataset_name == 'cifar10':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 100
    elif dataset_name == 'mnist':
        num_classes = 10
    else:
        raise NotImplementedError
    return num_classes


def GPU_warmup(device, in_shape=(3,224,224), warmup_iterations=20):
    # Load the pre-trained model
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    # model = models.__dict__[model_name](pretrained=True)
    model.to(device)
    model.eval()

    dummy_data = torch.randn(1, in_shape[0], in_shape[1], in_shape[2], dtype=torch.float).to(device)

    # Perform warm-up iterations
    for _ in range(warmup_iterations):
        _ = model(dummy_data)

    del dummy_data
    del model


