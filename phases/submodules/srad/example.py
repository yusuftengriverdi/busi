import cv2
import matplotlib.pyplot as plt
import numpy as np
from srad import *

Image = cv2.imread('data/benign/benign (8).png',1)
image = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
img   = np.array(image, float)

img_after = srad(img,Iterations = 200)
plt.subplot(1,2,1)
plt.imshow(img,cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(img_after,cmap = 'gray')
plt.show()
