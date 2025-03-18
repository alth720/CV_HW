import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt

img = cv.imread('mistyroad.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
t, bin_img = cv.threshold(gray, 127, 255, cv.THRESH_BINARY+cv.THRESH_OTSU) # 이진화 
print('오츄 알고리즘이 찾은 최적 임곗값=', t)

h = cv.calcHist([bin_img], [0], None, [256], [0, 256]) # 히스토그램 계산 

cv.imshow('Binary Image', bin_img)

plt.plot(h, color = 'r', linewidth = 1), plt.show()

cv.waitKey()
cv.destroyAllWindows()