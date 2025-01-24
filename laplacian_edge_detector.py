import numpy as np

def conv2d(image, kernel):
    """
    Perform 2D convolution on the given image using the provided kernel.
    
    Parameters:
        image (numpy.ndarray): Input grayscale image.
        kernel (numpy.ndarray): Convolution kernel.
    
    Returns:
        numpy.ndarray: Convolved image.
    """
    m, n = kernel.shape
    if m == n:
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y, x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m] * kernel)
    return new_image
