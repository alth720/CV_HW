import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt

img = cv.imread('IMG/mistyroad.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

t, bin_img = cv.threshold(gray, 127, 255, cv.THRESH_BINARY+cv.THRESH_OTSU) # 이진화 
print('오츄 알고리즘이 찾은 최적 임곗값=', t)

h1 = cv.calcHist([gray], [0], None, [256], [0, 256]) # 히스토그램 계산
h2 = cv.calcHist([bin_img], [0], None, [256], [0, 256]) # 히스토그램 계산 

cv.imshow('Binary Image', bin_img)

plt.plot(h1, color = 'b', linewidth = 1), plt.show()
plt.plot(h2, color = 'r', linewidth = 1), plt.show()

cv.waitKey()
cv.destroyAllWindows()
