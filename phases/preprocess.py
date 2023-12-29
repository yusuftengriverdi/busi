try: 
    from .submodules.srad import srad
except ImportError as e:
    from submodules.srad import srad
import numpy as np
import cv2

def apply_clahe(image, clip_limit=2.0, grid_size=(5, 5)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or color).
        clip_limit (float): Threshold for contrast limiting.
        grid_size (tuple): Size of grid for histogram equalization.

    Returns:
        numpy.ndarray: Image after applying CLAHE.
    """
    # Convert image to grayscale if it's in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced_image = clahe.apply(image)

    return enhanced_image

def apply_histogram_equalization(image):
    """
    Apply Histogram Equalization to enhance image contrast.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or color).

    Returns:
        numpy.ndarray: Image after applying Histogram Equalization.
    """
    # Convert image to grayscale if it's in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(image)

    return equalized_image

def preprocess(image_in):
    """
    Preprocess the input image by denoising and enhancing contrast.

    Parameters:
        image_in (numpy.ndarray): Input image (grayscale or color).

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    # Denoise the input image using the srad function (assumed to be available)
    img = srad.srad(image_in, Iterations=200)
    
    # Apply histogram equalization for further contrast enhancement
    img = apply_histogram_equalization(img.astype(np.uint8))

    return img

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    #1\Read the image
    Image = cv2.imread('data/malignant/malignant (11).png',1)
    image = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
    img   = np.array(image, float)

    img_after = preprocess(img)
    plt.subplot(1,2,1)
    plt.imshow(img,cmap = 'gray')
    plt.subplot(1,2,2)
    plt.imshow(img_after,cmap = 'gray')
    plt.show()
