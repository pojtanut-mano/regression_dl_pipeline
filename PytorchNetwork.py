import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchviz import make_dot

class Net(nn.Module):
    """Create neural network placeholder

    using Pytorch library to create neural network with the first layer (input layer) of specified size,
    2 hidden layers with 4 and 8 hidden nodes respectively and output node of size 1 being scaled by ReLU
    as a regression prediction result

    Args:
        input_size (int): dimension of input data
        output_size (int): dimension of output (default=1)

    Returns:
        (torch.tensor): output torch tensor with attached computational graph
    """
    def __init__(self, input_size, output_size=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, output_size)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        # First layer
        x = self.relu(self.fc1(x))
        # x = self.dropout1(x)

        # Second layer
        x = self.relu(self.fc2(x))
        # x = self.dropout2(x)

        # Output layer
        x = self.fc3(x)
        return x


def within_15(y_true, y_pred):
    """Return whether prediction is within 15 percent error of target

    Return 1 if the absolute percentage of error of predicted target is within 15 percent of true label

    Args:
        y_true (torch.tensor): true labels
        y_pred (torch.tensor): predicted labels

    Returns:
        None
    """
    return int(torch.sum(torch.abs(y_true - y_pred) * 100. / y_true <= 15)) / len(y_true)


def train(args, model, device, train_loader, optimizer, loss_fn, epoch):
    """Process data into network and perform back propagation to train it

    For each mini batch, run data through network and compute loss to adjust the weight of each neuron in network
    the number of epoch is specified in args, and there is a choice to dry run the network

    Args:
        args: argument in argparse module
        model (Net): Pytorch neural network that is already created by Net class as a sub-class of nn.Module
        device: specify GPU used to train network if not specify will use CPU instead
        train_loader: Pytorch DataLoader containing features and labels
        optimizer: Specify Pytorch optimizer object to be used as network optimizer
        loss_fn: Specify Pytorch loss function to calculate loss for each mini batch
        epoch: number of epoch that will be used to train the model
    Returns:
        None
    """
    model.train()
    data_count = 0
    count = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        model.zero_grad()

        data, target = data.to(device), target.to(device)
        data.requires_grad_(True)
        output = model.forward(data)

        loss = loss_fn(output, target)
        loss.backward()

        # Check result
        # print(output[:2], target[:2])
        # print('\n')

        # Check gradients
        # for name, param in model.named_parameters():
        #     print(name, param.grad.abs().sum())
        # print('\n')

        optimizer.step()

        data_count += len(data)
        if (count+1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                epoch, data_count, len(train_loader.dataset),
                100. * data_count / len(train_loader.dataset), loss.item()
            ))
            if args.dry_run:
                break
        count += 1


def test(model, device, loss_fn, test_loader):
    """Use test data to evaluate the performance of trained model

      run data through model and compute loss, then report the accuracy

       Args:
           model (Net): Pytorch neural network that is already created by Net class as a sub-class of nn.Module
           device: specify GPU used to train network if not specify will use CPU instead
           loss_fn: Specify Pytorch loss function to calculate loss for each mini batch
           test_loader: Pytorch DataLoader containing features and labels
       Returns:
           None
       """
    model.eval()
    test_loss = 0
    acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            acc += within_15(target, output)
        print("Average test accuracy: {:.2f} %\tloss: {:.3f}\n".format(acc/len(test_loader), test_loss / len(test_loader)))



