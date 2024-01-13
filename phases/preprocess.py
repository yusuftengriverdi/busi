try: 
    from .submodules.srad import srad
except ImportError as e:
    from submodules.srad import srad
import numpy as np
import cv2
import torch 
import os, glob
from tqdm import tqdm
import datetime
import shutil
# Reference of preprocessing: 

# @inproceedings{inproceedings,
# author = {Almajalid, Rania and Shan, Juan and Du, Yaodong and Zhang, Ming},
# year = {2018},
# month = {12},
# pages = {1103-1108},
# title = {Development of a Deep-Learning-Based Method for Breast Ultrasound Image Segmentation},
# doi = {10.1109/ICMLA.2018.00179}
# }

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
    exception_flag = False
    if len(image.shape) == 3:
        try: 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            image = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
            orig_shape = image.shape
            exception_flag = True

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(image)
    if exception_flag:
        equalized_image = torch.tensor(equalized_image).view(orig_shape)

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
    img = srad.srad(image_in, Iterations=50) #change 200 to 50 
     
    # Apply histogram equalization for further contrast enhancement
    img = apply_clahe(img.astype(np.uint8)) #change hist_eq to clahe

    return img

def preprocess_all(args):

    for label in ['benign', 'malignant', 'normal']:
        path_pattern = os.path.join(args.ROOT, f'{label}/*.png')
        file_paths = glob.glob(path_pattern)
        for file_path in tqdm(file_paths, desc=f'Preprocessing {label} cases'):
            filename = os.path.basename(file_path).split('.')[0]
            target_path = f'data/prep2/{label}/{filename}.png'
            if not os.path.exists(target_path):
                if not '_mask' in file_path:
                    img = cv2.imread(file_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = preprocess(img)

                    cv2.imwrite(target_path, img)
                else:
                    # Copy the file to the destination directory
                    shutil.copy(file_path, target_path)
                continue
            else:
                continue
    
    # Write information to the log file
    with open('data/prep/cache.log', mode='w') as log_file:
        log_file.write("len(file_paths)")
        log_file.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


    pass

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
