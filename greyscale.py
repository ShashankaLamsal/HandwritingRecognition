import cv2
import numpy as np

def apply_greyscale(image_path):
    """
    Convert the given image to greyscale and return it as a NumPy array.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: Greyscale image.
    """
    # Read the image in color
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image format or path")

    # Convert the image to greyscale
    greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return greyscale_image
