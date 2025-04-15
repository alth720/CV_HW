import numpy as np
import cv2 as cv

img = cv.imread('IMG/rose.png')

rows, cols = img.shape[:2]

center = (cols // 2, rows // 2)  # 중심 좌표
angle = 45  # 회전 각도
scale = 1.5  # 1.5배 확대

rot_matrix = cv.getRotationMatrix2D(center, angle, scale) # 회전 변환 행렬 생성 

rot_img = cv.warpAffine(img, rot_matrix, (cols, rows), flags=cv.INTER_LINEAR) # 회전 및 확대 

img_small = cv.resize(img, dsize = (0, 0), fx = 0.5, fy = 0.5)
rotImg_small = cv.resize(rot_img, dsize = (0, 0), fx = 0.5, fy = 0.5)

imgs = np.hstack((img_small, rotImg_small))

cv.imshow('Result', imgs)
cv.waitKey()
cv.destroyAllWindows()
