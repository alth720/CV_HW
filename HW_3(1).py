import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt

img = cv.imread('edgeDetectionImage.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

grad_x = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize = 3) # 소벨 연산자 적용 
grad_y = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize = 3)

magnitude = cv.magnitude(grad_x, grad_y) # 에지 강도 계산

edge_strength = cv.convertScaleAbs(magnitude) # 절댓값을 취해 양수 영상으로 변환 

# matplotlib로 시각화
plt.figure(figsize=(10, 5))

# 원본 이미지 (그레이스케일)
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# 에지 강도 이미지
plt.subplot(1, 2, 2)
plt.imshow(edge_strength, cmap='gray')
plt.title('Edge Strength')
plt.axis('off')

plt.tight_layout()
plt.show()