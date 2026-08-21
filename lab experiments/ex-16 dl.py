# OTSU THRESHOLDING AND DILATION

import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# Download image
url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
urllib.request.urlretrieve(url, "image.jpg")

# Read image
img = cv2.imread("image.jpg")

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Otsu Thresholding
_, thresh = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Create kernel
kernel = np.ones((2, 2), np.uint8)

# Morphological Closing
closing = cv2.morphologyEx(
    thresh,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

# Dilation
dilation = cv2.dilate(
    closing,
    kernel,
    iterations=3
)

# Display all results
plt.figure(figsize=(10, 7))

# 1. Original Image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# 2. Grayscale Image
plt.subplot(2, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

# 3. Otsu Threshold
plt.subplot(2, 2, 3)
plt.imshow(thresh, cmap="gray")
plt.title("Otsu Threshold")
plt.axis("off")

# 4. Dilation
plt.subplot(2, 2, 4)
plt.imshow(dilation, cmap="gray")
plt.title("Dilation")
plt.axis("off")

plt.tight_layout()
plt.show()
