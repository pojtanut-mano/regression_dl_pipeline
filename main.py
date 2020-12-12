from config import config
from Dataset import Dataset
from PytorchNetwork import Net, train, test
from preprocessing import PreProcessing
from datetime import datetime
import argparse
import numpy as np
import os
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, dataloader


def create_dir(dir_name):
    """
    Create directory with name dir_name

    :param
        dir_name (str) - name of the path
    :return:
        None
    """
    os.mkdir(dir_name)


def main():
    parser = argparse.ArgumentParser(description="Pytorch pipeline for regression")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default = 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # Create directory for each pass
    dir_name = datetime.now().strftime('%Y-%d-%m %H.%M.%S')
    create_dir(dir_name)

    # Prepare data
    holder = Dataset(config)
    X, y = holder.get_dataset(return_X_y=config['return_X_y'])

    prep = PreProcessing(X, y, config, dir_name)
    X_train, X_test, y_train, y_test = prep.get_dataset()

    # torch set up
    torch.manual_seed(config['seed'])
    if not args.no_cuda and not torch.cuda.is_available():
        print('No CUDA device detected...\nSwitching to CPU')
        device = torch.device('cpu')
    else:
        print('Using CUDA device {}\n'.format(torch.cuda.get_device_name()))
        device = torch.device('cuda')

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Convert to torch
    X_train_torch, X_test_torch, y_train_torch, y_test_torch = \
        torch.from_numpy(X_train.values).to(device).float(), torch.from_numpy(X_test.values).to(device).float(), \
        torch.from_numpy(y_train.values).to(device).float(), torch.from_numpy(y_test.values).to(device).float()

    # Define dataset
    train_ds = TensorDataset(X_train_torch, y_train_torch)
    test_ds = TensorDataset(X_test_torch, y_test_torch)

    # Define dataloader for both train and test
    train_loader = torch.utils.data.DataLoader(train_ds, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)

    # Define model
    model = Net(X_train.shape[1], 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loss_fn = nn.L1Loss()
    test_loss_fn = nn.L1Loss(reduction='sum')

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, train_loss_fn, epoch)
        test(model, device, test_loss_fn, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(dir_name, ''))


if __name__ == '__main__':
    main()
