import csv
import math
import os
import shutil
from typing import Any, Dict, List, NamedTuple, TextIO, Tuple

import numpy
import torch
import torchvision
from numpy import ndarray
from sklearn import metrics
from torch import LongTensor, Tensor

import captcha_recognizer
from captcha_dataset import TestCaptchaDataset
from captcha_recognizer import (
	CaptchaRecognizer,
	ModelDto,
	check_presence_and_index_to_dirpath,
	save_error_symbol_image_and_prediction)


class TestResults(NamedTuple):
	loss: float
	accuracy: float
	labels: List[str]
	targets: List[ndarray]
	predictions: List[ndarray]


def test(
		device: torch.device,
		model: torch.nn.Module,
		data_loader: torch.utils.data.DataLoader,
		captcha_alphabet_map: Dict[str, int]
	) -> TestResults:
	model.eval()

	loss_sum = 0.0
	assessed_samples = 0
	correct_predictions_count = 0
	labels: List[str] = []
	predictions: List[ndarray] = []
	targets: List[ndarray] = []
	with torch.no_grad():
		for data, label in data_loader:
			data: Tensor = data[0].to(device)
			target: LongTensor = captcha_recognizer.to_target(label[0], captcha_alphabet_map).to(device)

			output = model(data)

			if len(output) != len(target):
				continue

			loss_sum += torch.nn.functional.nll_loss(output, target, reduction="sum").item()
			assessed_samples += 1

			# get the index of the max log-probability
			sample_prediction = output.argmax(1, True).squeeze()
			correct_predictions_count += int(torch.equal(sample_prediction, target))

			labels.append(label[0])
			targets.append(target.numpy())
			predictions.append(sample_prediction.numpy())

	shutil.rmtree(os.path.join("error", "all-captcha"), ignore_errors = True)

	average_loss = loss_sum / assessed_samples if assessed_samples > 0 else math.nan
	accuracy = 100 * correct_predictions_count / len(data_loader.dataset)
	return TestResults(average_loss, accuracy, labels, targets, predictions)


def write_incorrect_prediction_images(epoch: int, test_results: TestResults, labels_map: List[str]) -> None:
	os.makedirs(os.path.join("error", f"epoch-{epoch}"), exist_ok = True)
	for label, symbol_predictions in zip(test_results.labels, test_results.predictions):
		dir_path = check_presence_and_index_to_dirpath(os.path.join("error", f"epoch-{epoch}", label))
		for i in range(len(symbol_predictions)):
			correct_symbol = label[i] if i < len(label) else ""
			if labels_map[symbol_predictions[i]] != correct_symbol:
				save_error_symbol_image_and_prediction(
					labels_map[symbol_predictions[i]],
					correct_symbol,
					i,
					dir_path)


def write_model_characteristics(epoch: int, test_results: TestResults, report_writer: Any) -> None:
	report_writer.writerow(["Epoch", epoch])
	report_writer.writerow(["Test loss", test_results.loss])
	report_writer.writerow(["Accuracy", test_results.accuracy])

	if len(test_results.targets) == 0 or len(test_results.predictions) == 0:
		report_writer.writerow(["AUC", math.nan])
		return

	fpr, tpr, _ = metrics.roc_curve(
		numpy.concatenate(test_results.targets),
		numpy.concatenate(test_results.predictions),
		pos_label = 2)
	report_writer.writerow(["AUC", metrics.auc(fpr, tpr)])


def write_classification_report(
		test_results: TestResults,
		captcha_alphabet_map: Dict[str, int],
		report_file: TextIO
	) -> None:
	report = metrics.classification_report(
		numpy.concatenate(test_results.targets),
		numpy.concatenate(test_results.predictions),
		labels = list(captcha_alphabet_map.values()),
		target_names = list(captcha_alphabet_map.keys()))
	report_file.write(report)


def write_roc_curve(test_results: TestResults, report_writer: Any) -> None:
	fpr, tpr, thresholds = metrics.roc_curve(
		numpy.concatenate(test_results.targets),
		numpy.concatenate(test_results.predictions),
		pos_label = 2)
	report_writer.writerow(["fpr"] + fpr.tolist())
	report_writer.writerow(["tpr"] + tpr.tolist())
	report_writer.writerow(["Thresholds"] + thresholds.tolist())


def main(
		device_type: str = "cpu",
		image_timesteps_count: int = 200,
		dataset_directory_name: str = "ds-1-test",
		model_directory_name: str = "models",
		report_directory_name: str = "reports",
		output_report_file_name: str = "report-{0}-{1}"):
	model_directory_file_names = next(os.walk(model_directory_name), (None, None, []))[2]
	model_file_paths: List[Tuple[int, str]] = []
	for file_name in model_directory_file_names:
		file_path = os.path.join(model_directory_name, file_name)
		dto: ModelDto
		epoch: int
		try:
			dto = torch.load(file_path)
			epoch = dto.epoch
		except:
			continue

		model_file_paths.append((epoch, file_path))
	model_file_paths.sort(key = lambda x: x[0])

	device: torch.device = torch.device(device_type)
	image_transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		# leave only one channel since they're all the same
		torchvision.transforms.Lambda(lambda x: x[0:1, :, :]),
		torchvision.transforms.Normalize((0.1307,), (0.3081,))])
	loader_parameters = {"num_workers": 1, "pin_memory": True} if device_type == "cuda" else {}
	test_dataset = TestCaptchaDataset(dataset_directory_name, transform = image_transform)
	test_loader = torch.utils.data.DataLoader(test_dataset, **loader_parameters)
	captcha_alphabet_map = {x[1]: x[0] for x in enumerate(CaptchaRecognizer.captcha_alphabet)}
	captcha_alphabet_labels_map = list(CaptchaRecognizer.captcha_alphabet)

	os.makedirs(report_directory_name, exist_ok = True)
	for epoch, model_file_path in model_file_paths:
		dto: ModelDto = torch.load(model_file_path)

		model = CaptchaRecognizer(image_timesteps_count, 1, 28, 28).to(device)
		model.load_state_dict(dto.model_state)
		test_results = test(device, model, test_loader, captcha_alphabet_map)
		write_incorrect_prediction_images(epoch, test_results, captcha_alphabet_labels_map)

		model_characteristics_file_path = os.path.join(
			report_directory_name,
			output_report_file_name.format(dto.epoch, "model") + ".csv")
		with open(model_characteristics_file_path, "wt", newline = "") as model_characteristics_file:
			model_characteristics_csv_writer = csv.writer(
				model_characteristics_file,
				dialect = "excel-tab",
				delimiter = ";")
			write_model_characteristics(epoch, test_results, model_characteristics_csv_writer)

		if len(test_results.targets) == 0 or len(test_results.predictions) == 0:
			continue

		classification_report_file_path = os.path.join(
			report_directory_name,
			output_report_file_name.format(dto.epoch, "classification") + ".txt")
		with open(classification_report_file_path, "wt", newline = "") as classification_report_file:
			write_classification_report(test_results, captcha_alphabet_map, classification_report_file)

		roc_curve_file_path = os.path.join(
			report_directory_name,
			output_report_file_name.format(dto.epoch, "roc-curve") + ".csv")
		with open(roc_curve_file_path, "wt", newline = "") as roc_curve_file:
			roc_curve_csv_writer = csv.writer(
				roc_curve_file,
				dialect = "excel-tab",
				delimiter = ";")
			write_roc_curve(test_results, roc_curve_csv_writer)


if __name__ == "__main__":
	device_type = "cuda" if torch.cuda.is_available() else "cpu"
	main(device_type)
