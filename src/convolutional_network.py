import os
import string
import time
from typing import Dict, List, Optional, TextIO, Tuple, Union

import norse
import norse.torch as snn
import numpy
import torch
import torch.utils.data
import torchvision
from fast_ctc_decode import beam_search
from torch import LongTensor, Tensor

black_color_threshold = 100 + 1
white_color_threshold = 200
color_map_with_no_black_and_white_shades = ([0] * black_color_threshold
	+ list(range(black_color_threshold, white_color_threshold))
	+ [255] * (256 - white_color_threshold))


class ConvolutionalMaxPooling(torch.nn.Sequential):
	def __init__(
			self,
			input_channels_count: int,
			output_channels_count: int,
			convolutional_kernel_size: Union[int, Tuple[int, int]],
			convolutional_stride: Union[int, Tuple[int, int]],
			convolutional_padding: Union[int, Tuple[int, int]] = 0,
			do_pooling: bool = True) -> None:
		layers: List[torch.nn.Module] = [
			torch.nn.Conv2d(
				input_channels_count,
				output_channels_count,
				convolutional_kernel_size,
				convolutional_stride,
				convolutional_padding,
				bias = False),
			torch.nn.BatchNorm2d(output_channels_count)]
		if do_pooling:
			layers.append(torch.nn.MaxPool2d(2, 2))
		super(ConvolutionalMaxPooling, self).__init__(*layers)


class FeatureExtractor(torch.nn.Sequential):
	def __init__(self, input_channels_count: int = 1, output_channels_count: int = 32) -> None:
		super(FeatureExtractor, self).__init__(
			ConvolutionalMaxPooling(input_channels_count, 4, 3, 1, 0, False),
			ConvolutionalMaxPooling(4, 8, 3, 1, 0, False),
			ConvolutionalMaxPooling(8, 16, 5, 1),
			ConvolutionalMaxPooling(16, 24, 3, 1, 0, False),
			ConvolutionalMaxPooling(24, output_channels_count, 3, 1, 0, False))

		self.input_channels_count = input_channels_count
		self.output_channels_count = output_channels_count

	@staticmethod
	def calculate_output_feature_map_width(image_width: int) -> int:
		return (image_width - 2 - 2 - 4) // 2 - 2 - 2

	@staticmethod
	def calculate_output_feature_map_height(image_height: int) -> int:
		return (image_height - 2 - 2 - 4) // 2 - 2 - 2

	@staticmethod
	def calculate_output_features_count(image_width: int, image_height: int) -> int:
		return (FeatureExtractor.calculate_output_feature_map_width(image_width)
			* FeatureExtractor.calculate_output_feature_map_height(image_height))


class ConvolutionalNetwork(torch.nn.Module):
	# Space for the "blank" symbol which represents absence of symbol at the given part of input
	captcha_alphabet: str = " " + string.digits + string.ascii_uppercase

	def __init__(self, timesteps_count: int, channels_count: int, image_width: int, image_height: int) -> None:
		super(ConvolutionalNetwork, self).__init__()

		self.input_features_count = image_width * image_height
		self.timesteps_count = timesteps_count

		self.constant_current_encoder = snn.ConstantCurrentLIFEncoder(timesteps_count)
		self.feature_extractor = FeatureExtractor(channels_count)
		self.feature_map_width = FeatureExtractor.calculate_output_feature_map_width(image_width)

		self.dropout = torch.nn.Dropout(0.15)
		self.lsnn0 = snn.LSNNRecurrentCell(
			self.feature_extractor.output_channels_count * FeatureExtractor.calculate_output_feature_map_height(image_height),
			256)
		self.lsnn1 = snn.LSNNRecurrentCell(256, 128)
		self.out = snn.LILinearCell(128, len(self.captcha_alphabet))

	def forward(self, images_batch: Tensor) -> Tensor:
		input_spikes = self.constant_current_encoder(
			# Flatten the images
			images_batch.view(-1, self.input_features_count))
		# Unflatten the images
		input_spikes = input_spikes.reshape(self.timesteps_count, *images_batch.shape)
		input_spikes = self.dropout(input_spikes)

		batch_states0: List[Optional[norse.torch.LSNNState]] = [None] * images_batch.shape[0]
		batch_states1: List[Optional[norse.torch.LSNNState]] = [None] * images_batch.shape[0]
		batch_states_out: List[Optional[norse.torch.LIState]] = [None] * images_batch.shape[0]
		out_voltages: List[Tensor] = []
		for timestep in range(self.timesteps_count):
			feature_map = self.feature_extractor(input_spikes[timestep])
			feature_map = feature_map.reshape(images_batch.shape[0], -1, feature_map.shape[-1])
			feature_map.transpose_(1, 2)
			
			batch_voltages: List[Tensor] = []
			for image_index in range(images_batch.shape[0]):
				voltages, batch_states0[image_index] = self.lsnn0(feature_map[image_index], batch_states0[image_index])
				voltages, batch_states1[image_index] = self.lsnn1(voltages, batch_states1[image_index])
				voltages, batch_states_out[image_index] = self.out(voltages, batch_states_out[image_index])
				batch_voltages.append(voltages.cpu())

			out_voltages.append(torch.stack(batch_voltages))

		# Gather voltages over all timesteps
		voltages = torch.max(torch.stack(out_voltages), 0).values
		voltages_length = torch.full((images_batch.shape[0],), self.feature_map_width, dtype = torch.long)

		return voltages, voltages_length


class ModelDto:
	def __init__(
			self,
			epoch: int,
			epochs_without_progress_in_row: int,
			model_state: dict,
			optimizer_state: dict,
			accuracy: float,
			best_test_loss: float):
		self.epoch = epoch
		self.epochs_without_progress_in_row = epochs_without_progress_in_row
		self.model_state = model_state
		self.optimizer_state = optimizer_state
		self.accuracy = accuracy
		self.best_test_loss = best_test_loss


class ModelParameters:
	def __init__(
			self,
			epoch: int,
			epochs_without_progress_in_row: int,
			model: torch.nn.Module,
			optimizer: torch.optim.Optimizer,
			accuracy: float,
			best_test_loss: float):
		self.epoch = epoch
		self.epochs_without_progress_in_row = epochs_without_progress_in_row
		self.model = model
		self.optimizer = optimizer
		self.accuracy = accuracy
		self.best_test_loss = best_test_loss


def train(
		device: torch.device,
		model: torch.nn.Module,
		ctc_loss_calculator: torch.nn.CTCLoss,
		optimizer: torch.optim.Optimizer,
		data_loader: torch.utils.data.DataLoader,
		targets: List[LongTensor],
		reports_count_per_epoch: int
	) -> List[float]:
	model.train()

	batches_per_report_count = (len(data_loader) + reports_count_per_epoch - 1) // reports_count_per_epoch
	report_batch_index = (len(data_loader) - 1) % batches_per_report_count
	losses: List[float] = []

	batch_processing_time_sum = 0
	for current_batch_index, (data, target_indices) in enumerate(data_loader):
		batch_start_time = time.perf_counter_ns()

		data: Tensor = data.to(device)
		target, target_length = to_target(targets, target_indices)

		optimizer.zero_grad()
		voltages, output_length = model(data)

		log_probabilities = torch.nn.functional.log_softmax(voltages, dim = 2)
		log_probabilities = log_probabilities.transpose(0, 1)
		loss: Tensor = ctc_loss_calculator.forward(
			log_probabilities,
			target,
			output_length,
			target_length)
		loss.backward()

		optimizer.step()

		losses.append(loss.item())

		batch_processing_time_sum += time.perf_counter_ns() - batch_start_time
		if current_batch_index % batches_per_report_count == report_batch_index:
			processed_batches_count = current_batch_index + 1
			current_progress_percent = processed_batches_count * 100 // len(data_loader)
			print(
				f"Loss: {loss.item():.7f} [{current_progress_percent}%, "
				f"{batch_processing_time_sum / 1e9:.1f} s elapsed]")

			if current_batch_index <= batches_per_report_count:
				average_batch_processing_time = batch_processing_time_sum / processed_batches_count / 1e6
				print(f"Average batch processing time: {average_batch_processing_time:.1f} ms")

	print(f"Average batch processing time: {batch_processing_time_sum / len(data_loader) / 1e6:.1f} ms")

	return losses


def test(
		device: torch.device,
		model: torch.nn.Module,
		ctc_loss_calculator: torch.nn.CTCLoss,
		data_loader: torch.utils.data.DataLoader,
		targets: List[LongTensor],
		labels: List[str],
		test_results_file: TextIO
	) -> Tuple[float, float]:
	model.eval()

	loss_sum = 0.0
	correct_predictions_count = 0
	ctc_decoding_time_sum = 0
	with torch.no_grad():
		for data, target_indices in data_loader:
			data: Tensor = data.to(device)
			target, target_length = to_target(targets, target_indices)

			voltages, output_length = model(data)

			log_probabilities = torch.nn.functional.log_softmax(voltages, dim = 2)
			log_probabilities = log_probabilities.transpose(0, 1)
			loss_sum += (ctc_loss_calculator
				.forward(
					log_probabilities,
					target,
					output_length,
					target_length)
				.item())

			batch_probabilities = torch.nn.functional.softmax(voltages, dim = 2)
			for probabilities, target_index in zip(batch_probabilities, target_indices):
				ctc_start_time = time.perf_counter_ns()
				output, path = beam_search(
					probabilities.numpy(),
					ConvolutionalNetwork.captcha_alphabet,
					12)
				ctc_decoding_time_sum += time.perf_counter_ns() - ctc_start_time

				target_label = labels[target_index]
				is_correct_prediction = output == target_label
				if is_correct_prediction:
					correct_predictions_count += 1

				test_results_file.write(f"{target_label},{output},{1 if is_correct_prediction else 0}\n")

	average_loss = loss_sum / len(data_loader)
	accuracy = 100 * correct_predictions_count / len(data_loader.dataset)
	print(f"Average CTC decoding time: {ctc_decoding_time_sum // len(data_loader.dataset) / 1e6:.1f} ms")
	print(f"Accuracy: {accuracy:.2f}%, test loss: {average_loss:.7f}")

	return average_loss, accuracy


def to_targets(
		target_map: Dict[str, int],
		alphabet_map: Dict[str, int]
	) -> List[LongTensor]:
	targets: List[LongTensor] = []
	for label, _ in sorted(target_map.items(), key = lambda x: x[1]):
		target = torch.as_tensor([alphabet_map[symbol] for symbol in label], dtype = torch.long)
		targets.append(target)
	return targets


def to_target(targets: List[LongTensor], target_indices: LongTensor) -> Tuple[LongTensor, LongTensor]:
	target = [targets[i.item()] for i in target_indices]
	target_length = [x.shape[0] for x in target]
	return torch.stack(target), torch.as_tensor(target_length, dtype = torch.long)


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
		dto.best_test_loss)


def save(file_path: str, parameters: ModelParameters) -> None:
	dto = ModelDto(
		parameters.epoch,
		parameters.epochs_without_progress_in_row,
		parameters.model.state_dict(),
		parameters.optimizer.state_dict(),
		parameters.accuracy,
		parameters.best_test_loss)
	torch.save(dto, file_path)


def main(
		device_type: str = "cpu",
		max_epoch_count: Optional[int] = None,
		early_stopping_epoch_count: Optional[int] = 4,
		batch_size: int = 32,
		test_dataset_fraction = 10000 / 70000,
		learning_rate: float = 2e-4,
		image_timesteps_count: int = 150,
		reports_count_per_epoch: int = 10,
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

	image_transform = torchvision.transforms.Compose([
		torchvision.transforms.Grayscale(),
		torchvision.transforms.Lambda(lambda image: image.point(color_map_with_no_black_and_white_shades)),
		torchvision.transforms.Lambda(lambda image: torchvision.transforms.functional.invert(image)),
		torchvision.transforms.Resize((28, 28)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.1307,), (0.3081,))])
	dataset = torchvision.datasets.ImageFolder("MNIST-like/", transform = image_transform)
	train_dataset, test_dataset = torch.utils.data.random_split(
		dataset,
		[1 - test_dataset_fraction, test_dataset_fraction])
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size)
	os.makedirs("test-results", exist_ok = True)

	model = ConvolutionalNetwork(image_timesteps_count, 1, 28, 28).to(device)
	ctc_loss_calculator = torch.nn.CTCLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

	captcha_alphabet_map = {x[1]: x[0] for x in enumerate(ConvolutionalNetwork.captcha_alphabet)}
	train_targets = to_targets(train_dataset.dataset.class_to_idx, captcha_alphabet_map)
	test_targets = to_targets(test_dataset.dataset.class_to_idx, captcha_alphabet_map)
	test_labels = list(x[0] for x in sorted(test_dataset.dataset.class_to_idx.items(), key = lambda x: x[1]))

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
		max_accuracy = loaded_model_parameters.accuracy
		epoch = loaded_model_parameters.epoch + 1
		epochs_without_progress_in_row = loaded_model_parameters.epochs_without_progress_in_row
	while ((max_epoch_count is None or epoch < max_epoch_count)
			and (early_stopping_epoch_count is None or epochs_without_progress_in_row < early_stopping_epoch_count)):
		epoch_start_time = time.perf_counter_ns()
		current_training_losses = train(
			device,
			model,
			ctc_loss_calculator,
			optimizer,
			train_loader,
			train_targets,
			reports_count_per_epoch)
		with open(f"test-results/epoch-{epoch}.csv", mode = "wt") as test_results_file:
			test_results_file.write("\"Target\",\"Prediction\",\"Is correct\"\n")
			test_loss, accuracy = test(
				device,
				model,
				ctc_loss_calculator,
				test_loader,
				test_targets,
				test_labels,
				test_results_file)
		epoch_end_time = time.perf_counter_ns()

		if best_test_loss is None or test_loss < best_test_loss:
			best_test_loss = test_loss
			epochs_without_progress_in_row = 0
		else:
			epochs_without_progress_in_row += 1

		if accuracy >= max_accuracy:
			max_accuracy = accuracy
			if output_model_file_name is not None:
				save(
					"models/" + output_model_file_name.format(epoch),
					ModelParameters(
						epoch,
						epochs_without_progress_in_row,
						model,
						optimizer,
						accuracy,
						best_test_loss))
		print(f"Epoch {epoch} is done in {(epoch_end_time - epoch_start_time) / 1e9:.1f} s")
		print()

		epoch += 1

	print(f"Max accuracy: {max_accuracy:.2f}%")


if __name__ == '__main__':
	device_type = "cuda" if torch.cuda.is_available() else "cpu"
	main(device_type)