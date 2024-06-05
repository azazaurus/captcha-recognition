from typing import Any, Callable, Optional

import PIL.Image
import cv2
from cv2.typing import MatLike
from torchvision.datasets import ImageFolder

from image_preprocessing import extend_to_square, extract_symbols, preprocess_image, resize


def opencv_to_pil(image: MatLike) -> PIL.Image.Image:
	converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return PIL.Image.fromarray(converted_image)


def default_captcha_transform(image: MatLike) -> MatLike:
	preprocessed_image = preprocess_image(image)
	symbol_regions = extract_symbols(preprocessed_image)
	if len(symbol_regions) != 1:
		# Bad sample but we can't do much here
		square_image = extend_to_square(preprocessed_image)
		resized_image = resize(square_image, 28, 28)
		return resized_image

	left, top, width, height = symbol_regions[0]
	symbol = preprocessed_image[top - 2:top + height + 2, left - 2:left + width + 2]
	square_symbol = extend_to_square(symbol)
	resized_symbol = resize(square_symbol, 28, 28)

	return resized_symbol


class CaptchaDataset(ImageFolder):
	def __init__(
			self,
			root: str,
			pre_transform: Optional[Callable[[MatLike], MatLike]] = default_captcha_transform,
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
