import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('JohnHancocksSignature.png', cv.IMREAD_UNCHANGED)

t, bin_img = cv.threshold(img[:,:,3], 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

b = bin_img[bin_img.shape[0]//2:bin_img.shape[0], 0:bin_img.shape[0]//2+1]

se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

dilation = cv.morphologyEx(b, cv.MORPH_DILATE, se)
erosion = cv.morphologyEx(b, cv.MORPH_ERODE, se)
opening = cv.morphologyEx(b, cv.MORPH_OPEN, se)
closing = cv.morphologyEx(b, cv.MORPH_CLOSE, se)

mp = np.hstack((b, dilation, erosion, opening, closing))
cv.imshow('Morphology', mp)

cv.waitKey()
cv.destroyAllWindows()