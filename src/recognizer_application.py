import sys
from typing import Optional

import cv2
import torch

from captcha_dataset import opencv_to_pil, test_captcha_transform
from captcha_recognizer import CaptchaRecognizer, ModelDto, load, create_symbol_image_transform


def main(
		model_file_name: str,
		device_type: Optional[str] = None,
		image_timesteps_count: int = 200
	) -> None:
	device_type = (device_type if device_type is not None else
		("cuda" if torch.cuda.is_available() else "cpu"))
	device: torch.device = torch.device(device_type)

	model = CaptchaRecognizer(image_timesteps_count, 1, 28, 28).to(device)
	load(model_file_name, model, None)
	labels_map = list(CaptchaRecognizer.captcha_alphabet)

	print("Type ! to exit")
	while True:
		input_image_file_name = input("CAPTCHA image file name: ")
		if input_image_file_name.startswith("!"):
			return

		input_image = cv2.imread(input_image_file_name)
		input_symbol_images = test_captcha_transform(input_image)
		symbol_image_transform = create_symbol_image_transform()
		symbol_images_to_recognize = (torch
			.stack(list(symbol_image_transform(opencv_to_pil(image)) for image in input_symbol_images))
			.to(device))
	
		with torch.no_grad():
			output = model(symbol_images_to_recognize)
			predictions = output.argmax(1, True)
			captcha_prediction = "".join(labels_map[prediction.item()] for prediction in predictions)
	
		print("Answer: " + captcha_prediction)
		print(flush = True)


if __name__ == "__main__":
	main(sys.argv[1])
