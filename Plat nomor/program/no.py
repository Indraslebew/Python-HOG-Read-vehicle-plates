import cv2
import numpy as np

# Fungsi untuk mencocokan pola menggunakan Template Matching
def match_pattern(image, template):
    # Ubah gambar menjadi grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Lakukan Template Matching
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)

    # Dapatkan lokasi pola dengan nilai korelasi tertinggi
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Dapatkan koordinat kotak yang mengelilingi pola
    template_width = gray_template.shape[1]
    template_height = gray_template.shape[0]
    top_left = max_loc
    bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

    # Gambar kotak di sekitar pola pada gambar asli
    matched_image = cv2.rectangle(image.copy(), top_left, bottom_right, (0, 255, 0), 2)

    # Tampilkan gambar yang telah dicocokkan polanya
    cv2.imshow("Hasil Cocok", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Hitung persentase kecocokan pola
    matching_percentage = max_val * 100

    return matching_percentage

# Baca gambar plat nomor kendaraan
plate_image = cv2.imread('17.jpeg')

# Baca gambar hasil HOG dari plat nomor kendaraan
hog_image = cv2.imread('17g.jpeg')

# Cocokkan pola menggunakan Template Matching
percentage = match_pattern(plate_image, hog_image)

# Tampilkan persentase kecocokan pola
print("Persentase Kecocokan Pola: {:.2f}%".format(percentage))
