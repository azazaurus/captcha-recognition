import random
from typing import List, Tuple

import norse
import norse.torch as snn
import numpy
import sklearn
import torch
import torchvision
from numpy import ndarray
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch import Tensor, LongTensor


def train(
        device: torch._C.device,
        model: norse.torch.SequentialState,
        optimizer: torch.optim.Optimizer,
        data_target_batches: List[Tuple[Tensor, LongTensor]]
    ) -> List[float]:
    model.train()
    current_batch_index = 0
    losses: List[float] = []

    for data, target in data_target_batches:
        data: Tensor = data.to(device)
        target: LongTensor = target.to(device)

        optimizer.zero_grad()
        output, _ = model(data)
        loss: Tensor = torch.nn.functional.nll_loss(output, target)
        loss.backward()
    
        optimizer.step()

        current_batch_index += 1
        losses.append(loss.item())
        print(f"loss: {loss.item():>7f} [{current_batch_index * 100 // len(data_target_batches)}%]")
        
    return losses


def test(
        device: torch._C.device,
        model: norse.torch.SequentialState,
        data_target_batches: List[Tuple[Tensor, LongTensor]]
    ) -> Tuple[float, float]:
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_target_batches:
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

    dataset_length = len(data_target_batches) * len(data_target_batches[0][0])
    test_loss /= dataset_length
    accuracy = 100.0 * correct / dataset_length
    print(f"Accuracy: {accuracy:>0.1f}%, test loss: {test_loss:>8f}")

    return test_loss, accuracy


def data_to_tensors(data: ndarray) -> List[Tensor]:
    data_transform = torchvision.transforms.Normalize((0.1307,), (0.3081,))
    return [data_transform(
            torch.from_numpy(
                numpy
                    .reshape(image, (1, 8, 8))
                    .astype(numpy.float32)))
        for image in data]


def to_data_target_pairs(data: ndarray, target: ndarray) -> List[Tuple[Tensor, int]]:
    data_tensors = data_to_tensors(data)
    return [(data_tensor, label) for data_tensor, label in zip(data_tensors, target)]


def split_into_batches(
        data_target_pairs: List[Tuple[Tensor, int]],
        batch_size: int,
        drop_last_uneven: bool = False
    ) -> List[Tuple[Tensor, LongTensor]]:
    if drop_last_uneven:
        data_target_pairs = data_target_pairs[:len(data_target_pairs) - len(data_target_pairs) % batch_size]

    data, target = zip(*data_target_pairs)
    numpy_data = numpy.stack([tensor.numpy() for tensor in data], axis = 0)
    data_batches = numpy.split(numpy_data, numpy.arange(batch_size, len(numpy_data), batch_size))
    target_batches = numpy.split(target, numpy.arange(batch_size, len(target), batch_size))

    return list(
        zip(
            (torch.from_numpy(batch) for batch in data_batches),
            (torch.as_tensor(batch, dtype = torch.long) for batch in target_batches)))


def main(
        device_type: str = "cpu",
        epoch_count: int = 100,
        batch_size: int = 32
    ) -> None:
    device: torch._C.device = torch.device(device_type)

    digits: sklearn.utils.Bunch = datasets.load_digits()
    train_data, test_data, train_target, test_target = train_test_split(
        digits["data"],
        digits["target"],
        test_size = 0.143)
    train_data_target_pairs: List[Tuple[Tensor, int]] = to_data_target_pairs(train_data, train_target)
    test_data_target_pairs: List[Tuple[Tensor, int]] = to_data_target_pairs(test_data, test_target)

    model = (norse.torch.SequentialState(
            torch.nn.Flatten(),
            norse.torch.LIFRecurrentCell(64, 16),
            torch.nn.Linear(16, 10),
            torch.nn.LogSoftmax(dim = 1))
        .to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-3)

    max_accuracy = 0.0
    random.seed()
    for epoch in range(epoch_count):
        random.shuffle(train_data_target_pairs)
        train_data_target_batches = split_into_batches(train_data_target_pairs, batch_size, True)

        random.shuffle(test_data_target_pairs)
        test_data_target_batches = split_into_batches(test_data_target_pairs, batch_size, True)

        current_training_losses = train(device, model, optimizer, train_data_target_batches)
        test_loss, accuracy = test(device, model, test_data_target_batches)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
        print(f"Epoch {epoch} is done")
        print()

    print(f"Max accuracy: {max_accuracy:>0.1f}%")


if __name__ == '__main__':
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    main(device_type)
