import math
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy
from cv2.typing import MatLike
from numpy import ndarray


class SymbolFilterParameters(NamedTuple):
	min_pixels_per_symbol: int = 0


def preprocess_image(image: MatLike) -> MatLike:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray_with_border = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
	smoothed = cv2.GaussianBlur(gray_with_border, (3, 3), 0)
	thresholded = (cv2
		.threshold(smoothed, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1])

	return thresholded


def crop(image: MatLike, cropping_contour: ndarray, padding: int = 0) -> MatLike:
	contour_mask = numpy.zeros((image.shape[0], image.shape[1], 1), dtype = numpy.uint8)
	cv2.drawContours(contour_mask, [cropping_contour], -1, (255), thickness = cv2.FILLED)

	image_with_contour_only = cv2.bitwise_and(image, image, mask = contour_mask)

	left, top, width, height = cv2.boundingRect(cropping_contour)
	cropped_contour = image_with_contour_only[top:top + height, left:left + width]

	if padding > 0:
		cropped_contour = cv2.copyMakeBorder(
			cropped_contour,
			padding,
			padding,
			padding,
			padding,
			cv2.BORDER_CONSTANT,
			None,
			0)

	return cropped_contour


def count_pixels(image: MatLike, contour: ndarray) -> int:
	contour_mask = numpy.zeros((image.shape[0], image.shape[1], 1), dtype = numpy.uint8)
	cv2.drawContours(contour_mask, [contour], -1, (255), thickness = cv2.FILLED)

	left, top, width, height = cv2.boundingRect(contour)
	return numpy.count_nonzero(contour_mask[top:top + height, left:left + width])


def get_center_of_region(region: Tuple[int, int, int, int]) -> Tuple[float, float]:
	left, top, width, height = region
	return (left + width) / 2, (top + height) / 2


def get_distance_between_regions(
		first: Tuple[int, int, int, int],
		second: Tuple[int, int, int, int]
	) -> float:
	first_center = get_center_of_region(first)
	second_center = get_center_of_region(second)
	return math.sqrt(
		(first_center[0] - second_center[0]) * (first_center[0] - second_center[0])
		+ (first_center[1] - second_center[1]) * (first_center[1] - second_center[1]))


def extract_symbol_contours(
		image: MatLike,
		symbol_filter_parameters: Optional[SymbolFilterParameters] = None
	) -> List[ndarray]:
	symbol_filter_parameters = (symbol_filter_parameters
		if symbol_filter_parameters is not None
		else SymbolFilterParameters())

	contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	symbol_contours: List[ndarray] = [
		x for x in contours
		if count_pixels(image, x) >= symbol_filter_parameters.min_pixels_per_symbol]
	# Sort the contours by theirs left borders
	symbol_contours.sort(key = lambda x: cv2.boundingRect(x)[0], reverse = False)
	return symbol_contours


def extend_to_square(image: MatLike) -> MatLike:
	height, width = image.shape
	if width == height:
		return image

	top_padding = max(0, (width - height) // 2)
	bottom_padding = max(0, width - height - top_padding)
	left_padding = max(0, (height - width) // 2)
	right_padding = max(0, height - width - left_padding)
	return cv2.copyMakeBorder(
		image,
		top_padding,
		bottom_padding,
		left_padding,
		right_padding,
		cv2.BORDER_CONSTANT,
		None,
		0)


def resize(image: MatLike, new_width: int, new_height: int) -> MatLike:
	height, width = image.shape
	if new_width == width and new_height == height:
		return image

	return cv2.resize(
		image,
		(new_width, new_height),
		interpolation = cv2.INTER_AREA if width < new_width or height < new_height else cv2.INTER_CUBIC)


def split_into_symbols_and_save(image_file_name: str):
	image = cv2.imread(image_file_name)
	processed_image = preprocess_image(image)
	symbol_filter_parameters = SymbolFilterParameters(
		processed_image.shape[0] * processed_image.shape[1] // 900)
	symbol_contours = extract_symbol_contours(processed_image, symbol_filter_parameters)
	for index, symbol_contour in enumerate(symbol_contours):
		letter = crop(processed_image, symbol_contour, 2)
		square_symbol = extend_to_square(letter)
		resized_symbol = resize(square_symbol, 28, 28)
		cv2.imwrite(f"{Path(image_file_name).stem}.{index}.png", resized_symbol)


def main():
	split_into_symbols_and_save("ds-1/1/1.28.png")
	split_into_symbols_and_save("ds-1/4/4.3784.png")
	split_into_symbols_and_save("ds-1/9/9.3336.png")
	split_into_symbols_and_save("ds-1/B/B.8.png")


if __name__ == "__main__":
	main()
