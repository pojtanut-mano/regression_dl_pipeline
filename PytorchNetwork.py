import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, output_size)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        # First layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Second layer
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Output layer
        x = self.fc3(x)
        return self.relu(x)


def within_15(y_true, y_pred):
    return int(torch.sum(torch.abs(y_true - y_pred) * 100. / y_true <= 15)) / len(y_true)


def train(args, model, device, train_loader, optimizer, loss_fn, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))
            if args.dry_run:
                break


def test(model, device, loss_fn, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            acc = within_15(target, output)
            print(acc)
        #     pred = output.argmax(dim=1, keepdim=True)
        #     correct += pred.eq(target.view_as(pred)).sum().item()
        #
        # test_loss /= len(test_loader.dataset)
        #
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(test_loader.dataset),
        #     100. * correct / len(test_loader.dataset)
        # ))


