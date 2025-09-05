import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

sapphires_img = Image.open("a1images/sapphire.jpg")

sapphires_img = np.array(sapphires_img)

sapphires_gray = cv.cvtColor(sapphires_img, cv.COLOR_RGB2HSV)[:, :, 1]
sapphires_blurred = cv.GaussianBlur(sapphires_gray, (5, 5), 0) 

threshold, sapphires_binary = cv.threshold(sapphires_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
sapphires_cleaned = cv.morphologyEx(sapphires_binary, cv.MORPH_CLOSE, kernel)


fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(sapphires_img)
axs[0].set_title("Original Image")
axs[0].axis('off')
axs[1].imshow(sapphires_gray, cmap='gray')
axs[1].set_title("Transformed Image")
axs[1].axis('off')
axs[2].imshow(sapphires_cleaned, cmap='gray')
axs[2].set_title(f"Otsu's Thresholding - {threshold:.2f}")
axs[2].axis('off')
plt.show()

num_labels, labels_im, stats, centroids = cv.connectedComponentsWithStats(sapphires_cleaned)

print(f"Number of sapphires: {num_labels - 1}")  # subtract 1 for background
areas = stats[1:, cv.CC_STAT_AREA]  
print("Areas of sapphires:", areas)