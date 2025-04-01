import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt

img = cv.imread('edgeDetectionImage.jpg')
imgS = cv.resize(img, dsize = (0, 0), fx = 0.5, fy = 0.5)
gray = cv.cvtColor(imgS, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray, 100, 200)

img_lines = imgS.copy()

# 허프 변환을 통한 직선 검출
lines = cv.HoughLinesP(
    canny,              # 입력 에지 이미지
    rho=1,              # 거리 해상도 (픽셀 단위)
    theta=np.pi/180,    # 각도 해상도 (라디안 단위)
    threshold=60,       # 직선으로 판단할 최소 교차점 수
    minLineLength=40,   # 직선으로 간주할 최소 길이
    maxLineGap=20       # 동일 선상의 점들을 연결할 최대 간격
)

# 직선 그리기
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

# matplotlib로 원본 이미지 + 직선 검출 이미지 나란히 표시
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(img_lines, cv.COLOR_BGR2RGB))
plt.title('Hough Lines Detected')
plt.axis('off')

plt.tight_layout()
plt.show()