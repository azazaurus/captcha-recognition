from typing import Any, Callable, List, Optional, Union

import PIL.Image
import cv2
import torch
from cv2.typing import MatLike
from torch import Tensor
from torchvision.datasets import ImageFolder

from image_preprocessing import (
	crop_with_padding,
	extend_to_square,
	extract_symbols,
	preprocess_image,
	resize)


def opencv_to_pil(image: MatLike) -> PIL.Image.Image:
	converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return PIL.Image.fromarray(converted_image)


def train_captcha_transform(image: MatLike) -> MatLike:
	preprocessed_image = preprocess_image(image)
	symbol_regions = extract_symbols(preprocessed_image)
	if len(symbol_regions) != 1:
		# Bad sample but we can't do much here
		square_image = extend_to_square(preprocessed_image)
		resized_image = resize(square_image, 28, 28)
		return resized_image

	symbol = crop_with_padding(preprocessed_image, symbol_regions[0], 2)
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


def test_captcha_transform(image: MatLike) -> List[MatLike]:
	preprocessed_image = preprocess_image(image)
	symbol_regions = extract_symbols(preprocessed_image)
	if len(symbol_regions) == 0:
		# Bad sample but we can't do much here
		square_image = extend_to_square(preprocessed_image)
		resized_image = resize(square_image, 28, 28)
		return [resized_image]

	symbols: List[MatLike] = []
	for symbol_region in symbol_regions:
		symbol = crop_with_padding(preprocessed_image, symbol_region, 2)
		square_symbol = extend_to_square(symbol)
		resized_symbol = resize(square_symbol, 28, 28)
		symbols.append(resized_symbol)

	return symbols


class TestCaptchaDataset(ImageFolder):
	def __init__(
			self,
			root: str,
			pre_transform: Callable[[MatLike], List[MatLike]] = test_captcha_transform,
			transform: Optional[Callable] = None
	) -> None:
		super().__init__(
			root,
			lambda images: self.captcha_transform(images, transform),
			lambda target_index: self.index_to_class_map[target_index],
			lambda file_name: self.load_file(file_name, pre_transform))
		self.index_to_class_map = list(
			class_ for (class_, _) in sorted(self.class_to_idx.items(), key = lambda x: x[1]))

	def load_file(self, file_name: str, pre_transform: Callable[[MatLike], List[MatLike]]) -> List[PIL.Image.Image]:
		image = cv2.imread(file_name)
		transformed_images = pre_transform(image)
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
