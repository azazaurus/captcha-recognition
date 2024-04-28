from typing import List, Tuple

import norse
import norse.torch as snn
import torch
import torch.utils.data
import torchvision
from torch import LongTensor, Tensor


class TwoLayerNetwork(torch.nn.Module):
	def __init__(self, timesteps_count: int) -> None:
		super(TwoLayerNetwork, self).__init__()

		self.timesteps_count = timesteps_count
		self.constant_current_encoder = snn.ConstantCurrentLIFEncoder(seq_length = timesteps_count)
		self.lif0 = snn.LIFCell()
		self.out = snn.LILinearCell(28 * 28, 10)

	def forward(self, image: Tensor) -> Tensor:
		image = image.reshape(*image.shape[:-3], -1)
		input_spikes = self.constant_current_encoder(image)

		state0 = None
		state_out = None
		out_voltages: List[Tensor] = []
		for timestep in range(self.timesteps_count):
			voltages, state0 = self.lif0(input_spikes[timestep], state0)
			voltages, state_out = self.out(voltages, state_out)
			out_voltages.append(voltages)

		voltages = torch.max(torch.stack(out_voltages), 0).values
		voltages = torch.nn.functional.log_softmax(voltages, dim = 1)
		
		return voltages


def train(
		device: torch._C.device,
		model: norse.torch.SequentialState,
		optimizer: torch.optim.Optimizer,
		data_loader: torch.utils.data.DataLoader,
		batch_size: int,
		reports_count_per_epoch: int
	) -> List[float]:
	model.train()
	batches_count = (len(data_loader.dataset) + batch_size - 1) // batch_size
	batches_per_report_count = (batches_count + reports_count_per_epoch - 1) // reports_count_per_epoch
	report_batch_index = (batches_count - 1) % batches_per_report_count
	losses: List[float] = []

	for current_index, (data, target) in enumerate(data_loader):
		data: Tensor = data.to(device)
		target: LongTensor = target.to(device)

		optimizer.zero_grad()
		output = model(data)
		loss: Tensor = torch.nn.functional.nll_loss(output, target)
		loss.backward()
	
		optimizer.step()

		losses.append(loss.item())
		
		if current_index % batches_per_report_count == report_batch_index:
			current_progress_percent = current_index * 100 // batches_count
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

			output = model(data)
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
		epoch_count: int = 50,
		batch_size: int = 32,
		image_timesteps_count: int = 100,
		reports_count_per_epoch: int = 5
	) -> None:
	device: torch._C.device = torch.device(device_type)

	torchvision.datasets.EMNIST.url = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
	train_loader = torch.utils.data.DataLoader(
		torchvision.datasets.EMNIST(
			root = ".",
			split = "digits",
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
			split = "digits",
			train = False,
			transform = torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
		batch_size = batch_size)

	model = TwoLayerNetwork(image_timesteps_count).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr = 2e-3)

	max_accuracy = 0.0
	for epoch in range(epoch_count):
		current_training_losses = train(device, model, optimizer, train_loader, batch_size, reports_count_per_epoch)
		test_loss, accuracy = test(device, model, test_loader)

		if accuracy > max_accuracy:
			max_accuracy = accuracy
		print(f"Epoch {epoch} is done")
		print()

	print(f"Max accuracy: {max_accuracy:>0.1f}%")


if __name__ == '__main__':
	device_type = "cuda" if torch.cuda.is_available() else "cpu"
	main(device_type)
