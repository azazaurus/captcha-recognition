from pathlib import Path
from typing import List, Tuple

import cv2
from cv2.typing import MatLike


def preprocess_image(image: MatLike) -> MatLike:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray_with_border = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
	thresholded = (cv2
		.threshold(gray_with_border, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1])

	return thresholded


def crop_with_padding(image: MatLike, cropping_area: Tuple[int, int, int, int], padding: int) -> MatLike:
	left, top, width, height = cropping_area
	return image[
		max(top - padding, 0):min(top + height + padding, image.shape[0]),
		max(left - padding, 0):min(left + width + padding, image.shape[1])]


def extract_symbols(image: MatLike) -> List[Tuple[int, int, int, int]]:
	contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	symbol_regions: List[Tuple[int, int, int, int]] = []
	for contour in contours:
		(left, top, width, height) = cv2.boundingRect(contour)

		# Compare the width and height of the contour to detect symbols that are
		# conjoined into one chunk
		if width / height > 1.25:
			# This contour is too wide to be a single symbol
			# Split it in half into two symbol regions
			half_width = width // 2
			symbol_regions.append((left, top, half_width, height))
			symbol_regions.append((left + half_width, top, half_width, height))
		else:
			# This is a normal symbol by itself
			symbol_regions.append((left, top, width, height))

	symbol_regions.sort(key = lambda x: x[0], reverse = False)
	return symbol_regions


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
	symbol_regions = extract_symbols(processed_image)
	for index, symbol_region in enumerate(symbol_regions):
		letter = crop_with_padding(processed_image, symbol_region, 2)
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
