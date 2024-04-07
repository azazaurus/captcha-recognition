import random
from typing import List, Tuple

import norse
import norse.torch as snn
import numpy
import sklearn
import torch
import torch.utils.data
import torchvision
from numpy import ndarray
from torch import Tensor, LongTensor


def train(
        device: torch._C.device,
        model: norse.torch.SequentialState,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader,
        batch_size: int
    ) -> List[float]:
    model.train()
    batch_count = (len(data_loader.dataset) + batch_size - 1) // batch_size
    report_batch_count = (batch_count + 10 - 1) // 10
    report_batch_index = (batch_count - 1) % report_batch_count
    losses: List[float] = []

    for current_batch_index, (data, target) in enumerate(data_loader):
        data: Tensor = data.to(device)
        target: LongTensor = target.to(device)

        optimizer.zero_grad()
        output, _ = model(data)
        loss: Tensor = torch.nn.functional.nll_loss(output, target)
        loss.backward()
    
        optimizer.step()

        losses.append(loss.item())
        
        if current_batch_index % report_batch_count == report_batch_index:
            current_progress_percent = current_batch_index * 100 // batch_count
            print(f"loss: {loss.item():>7f} [{current_progress_percent}%]")
        
    return losses


def test(
        device: torch._C.device,
        model: norse.torch.SequentialState,
        data_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data: Tensor = data.to(device)
            target: LongTensor = target.to(device)

            output, _ = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)
    print(f"Accuracy: {accuracy:>0.1f}%, test loss: {test_loss:>8f}")

    return test_loss, accuracy


def main(
        device_type: str = "cpu",
        epoch_count: int = 100,
        batch_size: int = 32
    ) -> None:
    device: torch._C.device = torch.device(device_type)

    torchvision.datasets.EMNIST.url = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.EMNIST(
            root = ".",
            split = "letters",
            train = True,
            download = True,
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size = batch_size,
        shuffle = True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.EMNIST(
            root = ".",
            split = "letters",
            train = False,
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size = batch_size)

    model = (norse.torch.SequentialState(
            torch.nn.Flatten(),
            norse.torch.LIFRecurrentCell(28 * 28, 32),
            torch.nn.Linear(32, 27),
            torch.nn.LogSoftmax(dim = 1))
        .to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-3)

    max_accuracy = 0.0
    for epoch in range(epoch_count):
        current_training_losses = train(device, model, optimizer, train_loader, batch_size)
        test_loss, accuracy = test(device, model, test_loader)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
        print(f"Epoch {epoch} is done")
        print()

    print(f"Max accuracy: {max_accuracy:>0.1f}%")


if __name__ == '__main__':
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    main(device_type)
