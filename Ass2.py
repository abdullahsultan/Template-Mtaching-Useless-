import cv2
import numpy as np

full_Image = cv2.imread("br.jpg")  # Loading full image
grey_full_Image = cv2.cvtColor(full_Image, cv2.COLOR_BGR2GRAY)  # Converting image to grey  scale
template = cv2.imread("ch.jpg", 0)  # Will load cropped image as grey scale
w, h = template.shape[::-1]  # Getting width and height of cropped image to draw a box around matching area in big pic
# -1 means opposite

resulting_matrix = cv2.matchTemplate(grey_full_Image, template, cv2.TM_CCORR_NORMED)  # This mathematical function
# will give us a
# matrix
# of full image with some values the
# value nearest to 1 will be most matching area of picture to template
print(resulting_matrix)
threshold = 0.99
locton = np.where(resulting_matrix >= threshold)  # will tell us how many values are nearest threshold to in matrix
print(locton)

# Looping to get best from matched templates
# zip(*locton) will iterate over all values it have (Unzipping)
for pt in zip(*locton[::-1]):
    cv2.rectangle(full_Image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)  # It will draw rectangle around matched
    # pt== start
# pt[0] + w, pt[1] + h means end of area # (0, 0, 255), 2 is width and color of rectangle

cv2.imshow("full_Image", full_Image)
cv2.waitKey(0)
cv2.destroyAllWindows()
