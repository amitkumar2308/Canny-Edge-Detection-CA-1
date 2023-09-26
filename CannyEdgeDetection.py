import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, feature
from scipy import ndimage


#for loading the image
image = io.imread('ct1.jpg')

downscale_factor = 2

# Perform datapreprocessing degrading resolution of picture
downscaled_image = ndimage.zoom(image, (1/downscale_factor, 1/downscale_factor, 1), order=1)

#converting image to grayscale
image_gray = color.rgb2gray(downscaled_image)

#apply gaussian blur to image (reducing noise)
blur_image = filters.gaussian(image_gray, sigma=1.0)

#calculate the gradient of the image using Sobel operator  

sobel_x = filters.sobel(blur_image, axis=1)  #Gradient is the rate of change of image intensity, and Sobel operator is a 3x3 kernel used to calculate the gradient of an image.
sobel_y = filters.sobel(blur_image, axis=0)

edges = feature.canny(blur_image, sigma=1.0, low_threshold=0.1, high_threshold=0.9) #Non-maximum suppression (NMS) is a technique used to thin out the edges detected in an image

#Display original iamge
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()

#display image with edges
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')
plt.show()
