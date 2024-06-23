import csv
import sys
from typing import Optional

import cv2
import torch

from captcha_dataset import opencv_to_pil, test_captcha_transform
from captcha_recognizer import CaptchaRecognizer, ModelDto, load, create_symbol_image_transform


def main(
		model_file_name: str,
		index_file_name: str,
		device_type: Optional[str] = None,
		image_timesteps_count: int = 200
	) -> None:
	device_type = (device_type if device_type is not None else
		("cuda" if torch.cuda.is_available() else "cpu"))
	device: torch.device = torch.device(device_type)

	model = CaptchaRecognizer(image_timesteps_count, 1, 28, 28).to(device)
	load(model_file_name, model, None)
	labels_map = list(CaptchaRecognizer.captcha_alphabet)

	with open(index_file_name, "rt", newline = "") as index_file:
		index_csv_reader = csv.reader(index_file)

		for captcha_file_name, correct_answer in index_csv_reader:
			print("CAPTCHA image file name: " + captcha_file_name)
			input_image = cv2.imread(captcha_file_name)
			input_symbol_images = test_captcha_transform(input_image)
			symbol_image_transform = create_symbol_image_transform()
			symbol_images_to_recognize = (torch
				.stack(list(symbol_image_transform(opencv_to_pil(image)) for image in input_symbol_images))
				.to(device))

			print("Correct answer: " + correct_answer)
			with torch.no_grad():
				output = model(symbol_images_to_recognize)
				predictions = output.argmax(1, True)
				captcha_prediction = "".join(labels_map[prediction.item()] for prediction in predictions)

			print("Prediction: " + captcha_prediction)
			print(
				"Prediction is correct"
					if captcha_prediction == correct_answer
					else "Prediction is NOT correct")
			print(flush = True)


if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])
