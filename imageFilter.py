import cv2
import numpy as np
import argparse
import os


# Mean filter function
def mean_filter(image):
    """
    Apply mean filter to an image.
    
    :param image: Input image.
    :param kernel_size: Size of the filter kernel (must be an odd number).
    :return: Processed image with mean filter applied.
    """
    kernel_size=13
    return cv2.blur(image, (kernel_size, kernel_size))

# Median filter function
def median_filter(image):
    """
    Apply median filter to an image.
    
    :param image: Input image.
    :param kernel_size: Size of the filter kernel (must be an odd number).
    :return: Processed image with median filter applied.
    """
    kernel_size=11
    return cv2.medianBlur(image, kernel_size)

def tophat_filtering(image):
    #Applies Top-Hat Filtering to enhance text while suppressing lines."""
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, tophat_kernel)

    # Apply Morphological Opening to extract horizontal lines
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, line_kernel, iterations=2)

    # Subtract detected lines from the original image
    processed = cv2.subtract(tophat, lines)
    return processed

def remove_lines_with_inpainting(image):
    #Removes horizontal lines using Morphological Opening and Inpainting."""
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, line_kernel, iterations=2)

    # Create a binary mask of the detected lines
    inpaint_mask = cv2.threshold(lines, 30, 255, cv2.THRESH_BINARY)[1]

    # Use inpainting to remove the lines
    processed = cv2.inpaint(image, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return processed
def remove_lines_adaptive(image):
    """Removes horizontal lines using adaptive morphological operations and inpainting."""
    # Detect horizontal lines
    height, width = image.shape
    line_kernel_size = max(10, width // 50)  # Adaptive kernel size based on image width
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_kernel_size, 1))

    lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, line_kernel, iterations=2)

    # Create a binary mask of detected lines
    _, inpaint_mask = cv2.threshold(lines, 30, 255, cv2.THRESH_BINARY)

    # Use inpainting to remove the lines
    processed = cv2.inpaint(image, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return processed

def hough(image):
    #Detects and removes horizontal lines using Hough Transform and inpainting."""
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=5)

    mask = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:  # Filter nearly horizontal lines
                cv2.line(mask, (x1, y1), (x2, y2), 255, 3)

    # Inpaint detected lines
    processed = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return processed
def fourier(image):
    #Removes horizontal lines using Fourier Transform filtering."""
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create a mask to remove horizontal frequencies
    rows, cols = image.shape
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[rows//2 - 2:rows//2 + 2, :] = 0  # Suppress horizontal components

    # Apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize and return result
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return img_back.astype(np.uint8)




def preprocess_image(image_path, output_path="processed.png", method="inpainting"):
    #Loads image, applies selected preprocessing method, and saves result.
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Choose the preprocessing method
    
    #processed = tophat_filtering(image)
    
    #processed = remove_lines_with_inpainting(image)
    processed=median_filter(image)
    processed1=mean_filter(image)
    # Apply Adaptive Thresholding to enhance text
    #processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)

    # Save processed image
    cv2.imwrite(output_path, processed)
    print(f"Processed image saved as: {output_path}")

    # Show results
    cv2.imshow("Original Image", image)
    cv2.imshow("Processed Image", processed)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess an image to remove lines.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--output", default="processed.png", help="Path to save the processed image")
    
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: File '{args.image_path}' not found!")
    else:
        preprocess_image(args.image_path, args.output)

# to run
#  python imageFilter.py uploads/kale.png
