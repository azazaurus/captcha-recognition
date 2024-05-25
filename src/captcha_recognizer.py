import os
import time
from typing import Dict, List, Optional, TextIO, Tuple

import norse.torch as snn
import numpy
import torch
import torch.utils.data
import torchvision
from torch import LongTensor, Tensor


class CaptchaRecognizer(torch.nn.Module):
	def __init__(self, timesteps_count: int, channels_count: int, image_width: int, image_height: int) -> None:
		super(CaptchaRecognizer, self).__init__()

		self.input_features_count = channels_count * image_width * image_height
		self.timesteps_count = timesteps_count

		self.constant_current_encoder = snn.ConstantCurrentLIFEncoder(timesteps_count)
		self.lif0 = snn.LIFCell()
		self.li0 = snn.LILinearCell(self.input_features_count, 2000)
		self.lif1 = snn.LIFCell()
		self.li1 = snn.LILinearCell(2000, 1500)
		self.lif2 = snn.LIFCell()
		self.li2 = snn.LILinearCell(1500, 1000)
		self.lif3 = snn.LIFCell()
		self.li3 = snn.LILinearCell(1000, 500)
		self.lif4 = snn.LIFCell()
		self.li4 = snn.LILinearCell(500, 100)
		self.lif5 = snn.LIFCell()
		self.out = snn.LILinearCell(100, 10)

	def forward(self, images_batch: Tensor) -> Tensor:
		batch_size = images_batch.shape[0]
		input_spikes = self.constant_current_encoder(
			# Flatten the images
			images_batch.view(batch_size, self.input_features_count))

		lif0_state = None
		li0_state = None
		lif1_state = None
		li1_state = None
		lif2_state = None
		li2_state = None
		lif3_state = None
		li3_state = None
		lif4_state = None
		li4_state = None
		lif5_state = None
		out_state = None
		timestep_outputs: List[Tensor] = []
		for timestep in range(self.timesteps_count):
			timestep_output, lif0_state = self.lif0(input_spikes[timestep], lif0_state)
			timestep_output, li0_state = self.li0(timestep_output, li0_state)
			timestep_output, lif1_state = self.lif1(timestep_output, lif1_state)
			timestep_output, li1_state = self.li1(timestep_output, li1_state)
			timestep_output, lif2_state = self.lif2(timestep_output, lif2_state)
			timestep_output, li2_state = self.li2(timestep_output, li2_state)
			timestep_output, lif3_state = self.lif3(timestep_output, lif3_state)
			timestep_output, li3_state = self.li3(timestep_output, li3_state)
			timestep_output, lif4_state = self.lif4(timestep_output, lif4_state)
			timestep_output, li4_state = self.li4(timestep_output, li4_state)
			timestep_output, lif5_state = self.lif5(timestep_output, lif5_state)
			timestep_output, out_state = self.out(timestep_output, out_state)
			timestep_outputs.append(timestep_output)

		# Gather output over all timesteps
		output = torch.max(torch.stack(timestep_outputs), 0).values
		output = torch.nn.functional.log_softmax(output, dim = 1)

		return output


class ModelDto:
	def __init__(
			self,
			epoch: int,
			epochs_without_progress_in_row: int,
			model_state: dict,
			optimizer_state: dict,
			accuracy: float,
			max_accuracy: float,
			best_test_loss: float):
		self.epoch = epoch
		self.epochs_without_progress_in_row = epochs_without_progress_in_row
		self.model_state = model_state
		self.optimizer_state = optimizer_state
		self.accuracy = accuracy
		self.max_accuracy = max_accuracy
		self.best_test_loss = best_test_loss


class ModelParameters:
	def __init__(
			self,
			epoch: int,
			epochs_without_progress_in_row: int,
			model: torch.nn.Module,
			optimizer: torch.optim.Optimizer,
			accuracy: float,
			max_accuracy: float,
			best_test_loss: float):
		self.epoch = epoch
		self.epochs_without_progress_in_row = epochs_without_progress_in_row
		self.model = model
		self.optimizer = optimizer
		self.accuracy = accuracy
		self.max_accuracy = max_accuracy
		self.best_test_loss = best_test_loss


def train(
		device: torch.device,
		model: torch.nn.Module,
		optimizer: torch.optim.Optimizer,
		data_loader: torch.utils.data.DataLoader,
		reports_count_per_epoch: int
	) -> List[float]:
	model.train()

	batches_per_report_count = (len(data_loader) + reports_count_per_epoch - 1) // reports_count_per_epoch
	report_batch_index = (len(data_loader) - 1) % batches_per_report_count
	losses: List[float] = []

	batch_processing_time_sum = 0
	for current_batch_index, (data, target) in enumerate(data_loader):
		batch_start_time = time.perf_counter_ns()

		data: Tensor = data.to(device)
		target: Tensor = target.to(device)

		optimizer.zero_grad()
		output = model(data)

		loss: Tensor = torch.nn.functional.nll_loss(output, target)
		loss.backward()

		optimizer.step()

		losses.append(loss.item())

		batch_processing_time_sum += time.perf_counter_ns() - batch_start_time
		if current_batch_index % batches_per_report_count == report_batch_index:
			processed_batches_count = current_batch_index + 1
			current_progress_percent = processed_batches_count * 100 // len(data_loader)
			print(
				f"Loss: {loss.item():.7f} [{current_progress_percent}%, "
				f"{batch_processing_time_sum / 1e9:.1f} s elapsed]",
				flush = True)

			if current_batch_index < batches_per_report_count:
				average_batch_processing_time = batch_processing_time_sum / processed_batches_count / 1e6
				print(f"Average batch processing time: {average_batch_processing_time:.1f} ms", flush = True)

	print(f"Average batch processing time: {batch_processing_time_sum / len(data_loader) / 1e6:.1f} ms")

	return losses


def test(
		device: torch.device,
		model: torch.nn.Module,
		data_loader: torch.utils.data.DataLoader,
		test_results_file: TextIO
	) -> Tuple[float, float]:
	model.eval()

	loss_sum = 0.0
	correct_predictions_count = 0
	with torch.no_grad():
		for data, target in data_loader:
			data: Tensor = data.to(device)
			target: Tensor = target.to(device)

			output = model(data)

			loss_sum += torch.nn.functional.nll_loss(output, target, reduction="sum").item()

			# get the index of the max log-probability
			predictions = output.argmax(1, True)
			for label, prediction in zip(target, predictions):
				is_correct_prediction = prediction.item() == label.item()
				if is_correct_prediction:
					correct_predictions_count += 1

				test_results_file.write(f"{label},{prediction},{1 if is_correct_prediction else 0}\n")

	average_loss = loss_sum / len(data_loader)
	accuracy = 100 * correct_predictions_count / len(data_loader.dataset)
	print(f"Accuracy: {accuracy:.2f}%, test loss: {average_loss:.7f}")

	return average_loss, accuracy


def to_target(labels: LongTensor, alphabet_map: Dict[str, int]) -> Tuple[LongTensor, LongTensor]:
	target = torch.as_tensor([alphabet_map[str(symbol.item())] for symbol in labels], dtype = torch.long)
	target_length = [1] * len(target)
	return target, torch.as_tensor(target_length, dtype = torch.long)


def load(
		file_path: str,
		model: torch.nn.Module,
		optimizer: torch.optim.Optimizer
	) -> ModelParameters:
	dto: ModelDto = torch.load(file_path)

	model.load_state_dict(dto.model_state)
	optimizer.load_state_dict(dto.optimizer_state)
	return ModelParameters(
		dto.epoch,
		dto.epochs_without_progress_in_row,
		model,
		optimizer,
		dto.accuracy,
		dto.max_accuracy,
		dto.best_test_loss)


def save(file_path: str, parameters: ModelParameters) -> None:
	dto = ModelDto(
		parameters.epoch,
		parameters.epochs_without_progress_in_row,
		parameters.model.state_dict(),
		parameters.optimizer.state_dict(),
		parameters.accuracy,
		parameters.max_accuracy,
		parameters.best_test_loss)
	torch.save(dto, file_path)


def main(
		device_type: str = "cpu",
		max_epoch_count: Optional[int] = None,
		early_stopping_epoch_count: Optional[int] = 10,
		batch_size: int = 8,
		learning_rate: float = 1e-3,
		image_timesteps_count: int = 150,
		reports_count_per_epoch: int = 1000,
		input_model_file_name: Optional[str] = None,
		output_model_file_name: Optional[str] = "model-{0}.pt",
		random_seed: Optional[int] = 1234
	) -> None:
	if random_seed is not None:
		numpy.random.seed(random_seed)
		torch.manual_seed(random_seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(random_seed)

	device: torch.device = torch.device(device_type)

	image_transform = torchvision.transforms.ToTensor()
	loader_parameters = {"num_workers": 1, "pin_memory": True} if device_type == "cuda" else {}
	train_dataset = torchvision.datasets.SVHN("SVHN", "train", image_transform, download = True)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, True, **loader_parameters)
	test_dataset = torchvision.datasets.SVHN("SVHN", "test", image_transform, download = True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, **loader_parameters)
	os.makedirs("test-results", exist_ok = True)

	model = CaptchaRecognizer(image_timesteps_count, 3, 32, 32).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

	loaded_model_parameters: Optional[ModelParameters] = None
	if input_model_file_name is not None:
		loaded_model_parameters = load("models/" + input_model_file_name, model, optimizer)
	if output_model_file_name is not None:
		os.makedirs("models", exist_ok = True)

	best_test_loss: Optional[float] = None
	max_accuracy = 0.
	epoch = 0
	epochs_without_progress_in_row = 0
	if loaded_model_parameters is not None:
		best_test_loss = loaded_model_parameters.best_test_loss
		max_accuracy = loaded_model_parameters.max_accuracy
		epoch = loaded_model_parameters.epoch + 1
		epochs_without_progress_in_row = loaded_model_parameters.epochs_without_progress_in_row
	while ((max_epoch_count is None or epoch < max_epoch_count)
			and (early_stopping_epoch_count is None or epochs_without_progress_in_row < early_stopping_epoch_count)):
		epoch_start_time = time.perf_counter_ns()
		current_training_losses = train(
			device,
			model,
			optimizer,
			train_loader,
			reports_count_per_epoch)
		with open(f"test-results/epoch-{epoch}.csv", mode = "wt") as test_results_file:
			test_results_file.write("\"Target\",\"Prediction\",\"Is correct\"\n")
			test_loss, accuracy = test(
				device,
				model,
				test_loader,
				test_results_file)
		epoch_end_time = time.perf_counter_ns()

		if best_test_loss is None or test_loss < best_test_loss:
			best_test_loss = test_loss
			epochs_without_progress_in_row = 0
		else:
			epochs_without_progress_in_row += 1

		max_accuracy = max(accuracy, max_accuracy)
		if output_model_file_name is not None:
			save(
				"models/" + output_model_file_name.format(epoch),
				ModelParameters(
					epoch,
					epochs_without_progress_in_row,
					model,
					optimizer,
					accuracy,
					max_accuracy,
					best_test_loss))
		print(f"Epoch {epoch} is done in {(epoch_end_time - epoch_start_time) / 1e9:.1f} s", flush = True)
		print()

		epoch += 1

	print(f"Max accuracy: {max_accuracy:.2f}%", flush = True)


if __name__ == '__main__':
	device_type = "cuda" if torch.cuda.is_available() else "cpu"
	main(device_type)
