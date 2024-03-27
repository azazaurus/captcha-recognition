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
        data_batches: List[Tensor],
        target_batches: List[LongTensor]
    ) -> List[float]:
    model.train()
    losses: List[float] = []

    for data, target in zip(data_batches, target_batches):
        data: Tensor = data.to(device)
        target: LongTensor = target.to(device)

        optimizer.zero_grad()
        output, _ = model(data)
        loss: Tensor = torch.nn.functional.nll_loss(output, target)
        loss.backward()
    
        optimizer.step()
    
        losses.append(loss.item())
        
    return losses


def test(
        device: torch._C.device,
        model: norse.torch.SequentialState,
        data_batches: List[Tensor],
        target_batches: List[LongTensor]
    ) -> Tuple[float, float]:
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in zip(data_batches, target_batches):
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

    dataset_length = len(data_batches) * len(data_batches[0])
    test_loss /= dataset_length
    accuracy = 100.0 * correct / dataset_length
    return test_loss, accuracy


def data_to_tensors(data: ndarray, batch_size: int) -> List[Tensor]:
    data_transform = torchvision.transforms.Normalize((0.1307,), (0.3081,))
    return [data_transform(
            torch.from_numpy(
                numpy
                    .reshape(batch, (batch.shape[0], 1, 8, 8))
                    .astype(numpy.float32)))
        for batch in split_into_batches(data, batch_size, True)]


def target_to_tensors(target: ndarray, batch_size: int) -> List[LongTensor]:
    return [torch.as_tensor(batch, dtype = torch.long)
        for batch in split_into_batches(target, batch_size, True)]


def split_into_batches(array: ndarray, batch_size: int, drop_last_uneven: bool = False) -> List[ndarray]:
    if drop_last_uneven:
        array = array[:len(array) - len(array) % batch_size]
    return numpy.split(array, numpy.arange(batch_size, len(array), batch_size))


def main(
        device_type: str = "cpu",
        epoch_count: int = 10,
        batch_size: int = 32
    ) -> None:
    device: torch._C.device = torch.device(device_type)

    digits: sklearn.utils.Bunch = datasets.load_digits()
    train_data, test_data, train_target, test_target = train_test_split(
        digits["data"],
        digits["target"],
        test_size = 0.143)
    train_data: List[Tensor] = data_to_tensors(train_data, batch_size)
    test_data: List[Tensor] = data_to_tensors(test_data, batch_size)
    train_target: List[LongTensor] = target_to_tensors(train_target, batch_size)
    test_target: List[LongTensor] = target_to_tensors(test_target, batch_size)

    model = (norse.torch.SequentialState(
            snn.Lift(torch.nn.Conv2d(1, 32, 3)),
            torch.nn.Flatten(2),
            snn.LIFRecurrent(36, 32),
            torch.nn.Flatten(1),
            snn.LILinearCell(1024, 10))
        .to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-3)

    max_accuracy = 0.0
    for epoch in range(epoch_count):
        current_training_losses = train(device, model, optimizer, train_data, train_target)
        test_loss, accuracy = test(device, model, test_data, test_target)
        
        if accuracy > max_accuracy:
            max_accuracy = accuracy
        print(f"Epoch {epoch} is done")
    
    print(f"Max accuracy: {max_accuracy}%")


if __name__ == '__main__':
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    main(device_type)
