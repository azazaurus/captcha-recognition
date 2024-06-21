import math
import os
import shutil
import string
import time
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

import norse.torch as snn
import numpy
import torch
import torch.utils.data
import torchvision
from torch import LongTensor, Tensor

from captcha_dataset import CaptchaDataset, TestCaptchaDataset


class CaptchaRecognizer(torch.nn.Module):
	captcha_alphabet: str = string.digits + string.ascii_uppercase

	def __init__(self, timesteps_count: int, channels_count: int, image_width: int, image_height: int) -> None:
		super(CaptchaRecognizer, self).__init__()

		self.input_features_count = channels_count * image_width * image_height
		self.fc_input_features_count = (64
			* (((image_width - 4) // 2 - 4) // 2)
			* (((image_height - 4) // 2 - 4) // 2))
		self.timesteps_count = timesteps_count

		self.constant_current_encoder = snn.ConstantCurrentLIFEncoder(timesteps_count)
		self.conv0 = torch.nn.Conv2d(channels_count, 32, 5, 1)
		self.lif0 = snn.LIFCell(
			snn.LIFParameters(
				method = "super",
				alpha = torch.tensor(100.0),
				v_th = torch.as_tensor(0.7)))
		self.conv1 = torch.nn.Conv2d(32, 64, 5, 1)
		self.lif1 = snn.LIFCell(
			snn.LIFParameters(
				method = "super",
				alpha = torch.tensor(100.0),
				v_th = torch.as_tensor(0.7)))
		self.fc0 = torch.nn.Linear(self.fc_input_features_count, 1024)
		self.lif2 = snn.LIFCell(snn.LIFParameters(method = "super", alpha = torch.tensor(100.0)))
		self.out = snn.LILinearCell(1024, len(CaptchaRecognizer.captcha_alphabet))

	def forward(self, images_batch: Tensor) -> Tensor:
		batch_size = images_batch.shape[0]
		input_spikes = self.constant_current_encoder(
			# Flatten the images
			images_batch.view(batch_size, self.input_features_count))
		input_spikes = input_spikes.reshape(self.timesteps_count, *images_batch.shape)

		lif0_state = None
		lif1_state = None
		lif2_state = None
		out_state = None
		timestep_outputs: List[Tensor] = []
		for timestep in range(self.timesteps_count):
			timestep_output = self.conv0(input_spikes[timestep])
			timestep_output, lif0_state = self.lif0(timestep_output, lif0_state)
			timestep_output = torch.nn.functional.max_pool2d(timestep_output, 2, 2)
			timestep_output *= 10
			timestep_output = self.conv1(timestep_output)
			timestep_output, lif1_state = self.lif1(timestep_output, lif1_state)
			timestep_output = torch.nn.functional.max_pool2d(timestep_output, 2, 2)
			timestep_output = timestep_output.view(batch_size, self.fc_input_features_count)
			timestep_output = self.fc0(timestep_output)
			timestep_output, lif2_state = self.lif2(timestep_output, lif2_state)
			timestep_output = torch.nn.functional.relu(timestep_output)
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
		captcha_alphabet_map: Dict[str, int],
		reports_count_per_epoch: int
	) -> List[float]:
	model.train()

	batches_per_report_count = (len(data_loader) + reports_count_per_epoch - 1) // reports_count_per_epoch
	report_batch_index = (len(data_loader) - 1) % batches_per_report_count
	losses: List[float] = []

	batch_processing_time_sum = 0
	for current_batch_index, (data, labels) in enumerate(data_loader):
		batch_start_time = time.perf_counter_ns()

		data: Tensor = data.to(device)
		target: LongTensor = to_target(labels, captcha_alphabet_map).to(device)

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


def check_presence_and_index_to_dirpath(dirpath_str: str) -> str:
	dirpath = Path(dirpath_str)
	directories_names = os.listdir(dirpath.parent)
	eponymous_directories = 1
	for x in directories_names:
		if x.startswith(dirpath.name):
			eponymous_directories += 1

	return dirpath_str + " (" + str(eponymous_directories) + ")"


def save_error_symbol_image_and_prediction(prediction, symbol_label, symbol_index, dir_path):
	src = os.path.join("error", "all-captcha", Path(dir_path).name, f"{symbol_index}.{symbol_label}.png")
	os.makedirs(dir_path, exist_ok = True)
	dst = os.path.join(dir_path, f"{symbol_index}.{symbol_label} ({prediction}).png")
	shutil.copy(src, dst)


def test(
		device: torch.device,
		model: torch.nn.Module,
		data_loader: torch.utils.data.DataLoader,
		captcha_alphabet_map: Dict[str, int],
		labels_map: List[str],
		epoch: int,
		test_results_file: TextIO,
		save_error_symbol_images = True
	) -> Tuple[float, float]:
	model.eval()
	if save_error_symbol_images:
		os.makedirs(os.path.join("error", f"epoch-{epoch}"), exist_ok = True)

	loss_sum = 0.0
	assessed_samples = 0
	correct_predictions_count = 0
	with torch.no_grad():
		for data, label in data_loader:
			data: Tensor = data[0].to(device)
			target: LongTensor = to_target(label[0], captcha_alphabet_map).to(device)

			output = model(data)

			if len(output) == len(target):
				loss_sum += torch.nn.functional.nll_loss(output, target, reduction="sum").item()
				assessed_samples += 1

			# get the index of the max log-probability
			predictions = output.argmax(1, True)
			captcha_prediction = "".join(labels_map[prediction.item()] for prediction in predictions)
			is_correct_prediction = captcha_prediction == label[0]
			if is_correct_prediction:
				correct_predictions_count += 1
			elif save_error_symbol_images:
				dir_path = check_presence_and_index_to_dirpath(os.path.join("error", f"epoch-{epoch}", label[0]))
				for i in range(len(predictions)):
					correct_symbol = label[0][i] if i < len(label[0]) else ""
					if labels_map[predictions[i].item()] != correct_symbol:
						save_error_symbol_image_and_prediction(labels_map[predictions[i].item()], correct_symbol, i, dir_path)

			test_results_file.write(f"{label[0]},{captcha_prediction},{1 if is_correct_prediction else 0}\n")

	shutil.rmtree(os.path.join("error", "all-captcha"), ignore_errors = True)

	average_loss = loss_sum / assessed_samples if assessed_samples > 0 else math.nan
	accuracy = 100 * correct_predictions_count / len(data_loader.dataset)
	print(f"Accuracy: {accuracy:.2f}%, test loss: {average_loss:.7f}")

	return average_loss, accuracy


def to_target(label: str, alphabet_map: Dict[str, int]) -> Tuple[LongTensor]:
	return torch.as_tensor([alphabet_map[symbol] for symbol in list(label)], dtype = torch.long)


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
		batch_size: int = 32,
		learning_rate: float = 2e-3,
		image_timesteps_count: int = 200,
		reports_count_per_epoch: int = 7500,
		input_model_file_name: Optional[str] = None,
		output_model_file_name: Optional[str] = "model-{0}.pt",
		random_seed: Optional[int] = 3456
	) -> None:
	if random_seed is not None:
		numpy.random.seed(random_seed)
		torch.manual_seed(random_seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(random_seed)

	device: torch.device = torch.device(device_type)

	image_transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		# leave only one channel since they're all the same
		torchvision.transforms.Lambda(lambda x: x[0:1, :, :]),
		torchvision.transforms.Normalize((0.1307,), (0.3081,))])
	loader_parameters = {"num_workers": 1, "pin_memory": True} if device_type == "cuda" else {}
	train_dataset = CaptchaDataset("ds-1/", transform = image_transform)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, True, **loader_parameters)
	test_dataset = TestCaptchaDataset("ds-1-test/", transform = image_transform)
	test_loader = torch.utils.data.DataLoader(test_dataset, **loader_parameters)
	os.makedirs("test-results", exist_ok = True)

	model = CaptchaRecognizer(image_timesteps_count, 1, 28, 28).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
	captcha_alphabet_map = {x[1]: x[0] for x in enumerate(CaptchaRecognizer.captcha_alphabet)}

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
			captcha_alphabet_map,
			reports_count_per_epoch)
		with open(f"test-results/epoch-{epoch}.csv", mode = "wt") as test_results_file:
			test_results_file.write("\"Target\",\"Prediction\",\"Is correct\"\n")
			test_loss, accuracy = test(
				device,
				model,
				test_loader,
				captcha_alphabet_map,
				list(CaptchaRecognizer.captcha_alphabet),
				epoch,
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
