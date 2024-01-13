# Apply superpixel to see if it works good.
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from preprocess import apply_clahe

def kmeans_roi(image, k=5):
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    labels = kmeans.labels_.reshape(image.shape[:2])
    largest_cluster_label = np.argmax(np.bincount(labels.flat))
    largest_cluster_mask = (labels == largest_cluster_label).astype(np.uint8)
    contours, _ = cv2.findContours(largest_cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return labels, (x, y, x + w, y + h), largest_cluster_mask

def superpixel_roi(image, num_segments=15):
    segments = slic(image, n_segments=num_segments, compactness=3)
    largest_superpixel_label = np.argmax(np.bincount(segments.flat))
    largest_superpixel_mask = (segments == largest_superpixel_label).astype(np.uint8)
    props = regionprops(largest_superpixel_mask)[0]
    y, x, y_end, x_end = props.bbox
    return segments, (x, y, x_end, y_end)

if __name__ == '__main__':
    from preprocess import preprocess

    image = cv2.imread("data/benign/benign (8).png")

    image = preprocess(image)

    image = cv2.GaussianBlur(image, (9, 9), 5)

    image = apply_clahe(image)

    # Apply thresholding
    ret, th = cv2.threshold(image,127, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Apply K-means clustering
    labels, kmeans_roi_bbox, largest = kmeans_roi(image)

    # Apply superpixel segmentation
    segments, superpixel_roi_bbox = superpixel_roi(image)

    # Visualize the results using Matplotlib subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 5))

    axs = axs.ravel()
    # Original image
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Blurred Image')

    axs[1].imshow(th)
    axs[1].set_title('Thresholded Image')

    # K-means ROI
    kmeans_roi_image = image.copy()
    cv2.rectangle(kmeans_roi_image, (kmeans_roi_bbox[0], kmeans_roi_bbox[1]), (kmeans_roi_bbox[2], kmeans_roi_bbox[3]), (0, 255, 0), 2)
    axs[2].imshow(cv2.cvtColor(kmeans_roi_image, cv2.COLOR_BGR2RGB))
    axs[2].imshow(labels)
    axs[2].set_title('K-means')

    axs[3].imshow(cv2.cvtColor(kmeans_roi_image, cv2.COLOR_BGR2RGB))
    axs[3].imshow(largest)
    axs[3].set_title('K-means Largest')

    # Superpixel ROI
    axs[4].imshow(segments)

    axs[4].set_title('Superpixel')

    plt.show()
