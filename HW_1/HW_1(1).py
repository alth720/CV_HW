import numpy as np
import cv2 as cv
import sys

img = cv.imread('IMG/soccer.jpg')
imgs_small = cv.resize(img, dsize = (0, 0), fx = 0.5, fy = 0.5)

if img is None :
    sys.exit('파일이 존재하지 않습니다.')

gray = cv.cvtColor(imgs_small, cv.COLOR_BGR2GRAY)
gray_img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

imgs = np.hstack((imgs_small,gray_img))
cv.imshow('Image Display', imgs)

cv.waitKey()
cv.destroyAllWindows()