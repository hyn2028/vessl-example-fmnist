import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

from pathlib import Path
from typing import List, Tuple

def train(dataloader, model, loss_fn, optimizer, args, writer, step):
    """
    Define training process.
    """

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(args.device), y.to(args.device)

        # calculate loss of prediction
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print learning state (each 100 batch)
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            writer.add_scalar('train/loss', loss, step + batch + 1)


def test(dataloader, model, loss_fn, args, writer, step) -> Tuple[float, float]:
    """
    Define testing process.
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(args.device), y.to(args.device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    writer.add_scalar('test/loss', test_loss, step)
    writer.add_scalar('test/acc', correct, step)

    return test_loss, correct


def prepare_data(root: Path, batch_size: int) -> Tuple[Dataset, Dataset, DataLoader, DataLoader, List[str]]:
    """
    Improt FashionMNIST dataset.
    Use local data, for testing VESSL dataset environment.
    """

    # improt FashionMNIST dataset.
    training_data = datasets.FashionMNIST(
        root=root,
        train=True,
        download=False,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root=root,
        train=False,
        download=False,
        transform=ToTensor(),
    )

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # generate dataloader from datasets
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    return training_data, test_data, train_dataloader, test_dataloader, class_names
    

def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    arr = torch.from_numpy(np.array(fig.canvas.renderer._renderer)) # H, W, RGBA
    arr = arr.permute(2, 0, 1) # RGBA, H, W
    arr = arr[:3] # RGB, H, W
    return arr