import numpy as np
import cv2 as cv
import sys

img = cv.imread('soccer.jpg')

if img is None :
    sys.exit('파일이 존재하지 않습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

imgs = np.hstack((img,gray_img))
cv.imshow('Image Display', imgs)

cv.waitKey()
cv.destroyAllWindows()