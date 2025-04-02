import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('mot_color70.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create(0, 3, 0.1, 10, 2) # SIFT 객체 생성성
kp, des = sift.detectAndCompute(gray, None)  # 키 포인트 검출 및 서술자 계산

gray = cv.drawKeypoints(gray, kp, None, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 키포인트 그리기 

# 원본 이미지
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# 특징점 검출 이미지
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))
plt.title("SIFT")
plt.axis('off')

plt.tight_layout()
plt.show()