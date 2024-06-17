import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import PIL.Image
import cv2
import torch
from cv2.typing import MatLike
from torch import Tensor
from torchvision.datasets import ImageFolder

from image_preprocessing import (
	SymbolFilterParameters,
	crop,
	extend_to_square,
	extract_symbol_contours,
	preprocess_image,
	resize)


def opencv_to_pil(image: MatLike) -> PIL.Image.Image:
	converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return PIL.Image.fromarray(converted_image)


def train_captcha_transform(image: MatLike) -> MatLike:
	preprocessed_image = preprocess_image(image)
	symbol_contours = extract_symbol_contours(preprocessed_image)
	if len(symbol_contours) != 1:
		# Bad sample but we can't do much here
		square_image = extend_to_square(preprocessed_image)
		resized_image = resize(square_image, 28, 28)
		return resized_image

	symbol = crop(preprocessed_image, symbol_contours[0], 2)
	square_symbol = extend_to_square(symbol)
	resized_symbol = resize(square_symbol, 28, 28)

	return resized_symbol


class CaptchaDataset(ImageFolder):
	def __init__(
			self,
			root: str,
			pre_transform: Optional[Callable[[MatLike], MatLike]] = train_captcha_transform,
			transform: Optional[Callable] = None
	) -> None:
		super().__init__(
			root,
			transform,
			lambda target_index: self.index_to_class_map[target_index],
			lambda file_name: self.load_file(file_name, pre_transform))
		self.index_to_class_map = list(
			class_ for (class_, _) in sorted(self.class_to_idx.items(), key = lambda x: x[1]))

	def load_file(self, file_name: str, pre_transform: Optional[Callable[[MatLike], MatLike]]) -> PIL.Image.Image:
		image = cv2.imread(file_name)
		transformed_image = pre_transform(image) if pre_transform is not None else image
		return opencv_to_pil(transformed_image)


def check_presence_and_index_to_dirpath(dirpath_str: str) -> str:
	dirpath = Path(dirpath_str)
	directories_names = os.listdir(dirpath.parent)
	eponymous_directories = 1
	for x in directories_names:
		if x.startswith(dirpath.name):
			eponymous_directories += 1

	return dirpath_str + " (" + str(eponymous_directories) + ")"


def save_symbol_image(image: MatLike, filename: str) -> None:
	os.makedirs(Path(filename).parent, exist_ok = True)
	cv2.imwrite(filename, image)


def test_captcha_transform(image: MatLike, label) -> List[MatLike]:
	preprocessed_image = preprocess_image(image)
	symbol_filter_parameters = SymbolFilterParameters(
		preprocessed_image.shape[0] * preprocessed_image.shape[1] // 900)
	symbol_contours = extract_symbol_contours(preprocessed_image, symbol_filter_parameters)
	if len(symbol_contours) == 0:
		# Bad sample but we can't do much here
		square_image = extend_to_square(preprocessed_image)
		resized_image = resize(square_image, 28, 28)
		return [resized_image]

	os.makedirs(os.path.join("error", "all-captcha"), exist_ok = True)
	dir_template = os.path.join("error", "all-captcha", label)
	dir_path = check_presence_and_index_to_dirpath(dir_template)

	symbols: List[MatLike] = []
	for i in range(0, len(symbol_contours)):
		symbol = crop(preprocessed_image, symbol_contours[i], 2)
		square_symbol = extend_to_square(symbol)
		resized_symbol = resize(square_symbol, 28, 28)
		symbols.append(resized_symbol)
		filename = os.path.join(dir_path, f"{i}.{label[i] if i < len(label) else ''}.png")
		save_symbol_image(resized_symbol, filename)

	return symbols


class TestCaptchaDataset(ImageFolder):
	def __init__(
			self,
			root: str,
			pre_transform: Callable[[MatLike, str], List[MatLike]] = test_captcha_transform,
			transform: Optional[Callable] = None
	) -> None:
		super().__init__(
			root,
			lambda images: self.captcha_transform(images, transform),
			lambda target_index: self.index_to_class_map[target_index],
			lambda file_name: self.load_file(file_name, pre_transform))
		self.index_to_class_map = list(
			class_ for (class_, _) in sorted(self.class_to_idx.items(), key = lambda x: x[1]))

	def load_file(self, file_name: str, pre_transform: Callable[[MatLike, str], List[MatLike]]) -> List[PIL.Image.Image]:
		image = cv2.imread(file_name)
		captcha_label = Path(file_name).parent.name
		transformed_images = pre_transform(image, captcha_label)
		return [opencv_to_pil(image) for image in transformed_images]

	def captcha_transform(
			self,
			images: Union[List[Any], Tensor],
			transform: Optional[Callable]
	) -> Union[List[Any], Tensor]:
		if transform is None:
			return images

		transformed_images = [transform(image) for image in images]
		return (torch.stack(transformed_images)
			if all(isinstance(image, Tensor) for image in transformed_images)
			else transformed_images)
