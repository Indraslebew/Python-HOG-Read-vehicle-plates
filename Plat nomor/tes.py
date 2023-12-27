import cv2
from skimage.feature import hog

# Load gambar
image = cv2.imread('gelap/20.jpg', cv2.IMREAD_GRAYSCALE)

# Konfigurasi HOG
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# Ekstraksi fitur HOG
features, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block, visualize=True)

# Tampilkan hasil ekstraksi fitur HOG
cv2.imshow('HOG Image', hog_image)
cv2.waitKey(0)
cv2.destroyAllWindows()