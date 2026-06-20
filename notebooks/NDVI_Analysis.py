# NDVI Analysis and Change Detection for Chitgar Lake
# This notebook processes Sentinel-2 imagery to analyze vegetation changes between 2023 and 2025.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the Images
# Note: Ensure the file names match the ones in your 'data' folder
img_2023 = cv2.imread('../data/photo_1781983102581.jpeg')
img_2025 = cv2.imread('../data/photo_1781983094316.jpeg')

# Convert BGR to RGB for correct display
img_2023 = cv2.cvtColor(img_2023, cv2.COLOR_BGR2RGB)
img_2025 = cv2.cvtColor(img_2025, cv2.COLOR_BGR2RGB)

# 2. Display the Original NDVI Maps
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.title("NDVI Map - May 2023")
plt.imshow(img_2023)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("NDVI Map - May 2025")
plt.imshow(img_2025)
plt.axis('off')

plt.show()

# 3. Calculate Difference Map
# We convert images to grayscale to find the intensity of NDVI values
gray_2023 = cv2.cvtColor(img_2023, cv2.COLOR_RGB2GRAY)
gray_2025 = cv2.cvtColor(img_2025, cv2.COLOR_RGB2GRAY)

# Calculate the pixel-wise difference
diff_map = cv2.absdiff(gray_2025, gray_2023)

plt.figure(figsize=(8, 8))
plt.title("NDVI Intensity Difference Map")
plt.imshow(diff_map, cmap='hot')
plt.colorbar(label='Change Intensity')
plt.axis('off')
plt.show()

# 4. Histogram Analysis
# Comparing the distribution of NDVI values between the two years
plt.figure(figsize=(12, 6))
plt.hist(gray_2023.ravel(), bins=256, color='blue', alpha=0.5, label='May 2023')
plt.hist(gray_2025.ravel(), bins=256, color='green', alpha=0.5, label='May 2025')
plt.title("NDVI Pixel Intensity Distribution (Histogram Comparison)")
plt.xlabel("Intensity Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 5. Summary Statistics
print(f"Mean Intensity 2023: {np.mean(gray_2023):.2f}")
print(f"Mean Intensity 2025: {np.mean(gray_2025):.2f}")
print(f"Standard Deviation 2023: {np.std(gray_2023):.2f}")
print(f"Standard Deviation 2025: {np.std(gray_2025):.2f}")
