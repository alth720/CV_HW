import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt

img1 = cv.imread('mot_color70.jpg')[190:350, 440:560]  # 버스를 크롭하여 모델 영상으로 사용 
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('mot_color83.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create(0, 3, 0.1, 10, 2)
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print('특징점 개수:', len(kp1), len(kp2))

# 인덱스 파라미터와 검색 파라미터 설정
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
search_params = dict(checks=90)

matcher = cv.FlannBasedMatcher(index_params, search_params)  # Flann 매처 생성

matches = matcher.knnMatch(des1, des2, k = 2)  # 매칭 계산

# Ratio Test 적용 (Lowe’s paper 기준)
T = 0.7
good_match = []
for nearest1, nearest2 in matches:
    if (nearest1.distance/nearest2.distance) < T:
        good_match.append(nearest1)

""" # BFMatcher 생성 (L2 norm은 SIFT에 적합)
bf = cv.BFMatcher(cv.NORM_L2)

# knnMatch로 k=2 설정 (두 개의 최근접 이웃)
matches = bf.knnMatch(des1, des2, k=2)

# Ratio Test 적용
T = 0.7
good_match = []
for nearest1, nearest2 in matches:
    if (nearest1.distance / nearest2.distance) < T:
        good_match.append(nearest1)
 """

img_match = cv.drawMatches(img1, kp1, img2, kp2, good_match, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 특징점 매칭 이미지
plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
plt.title("FLANN")
plt.axis('off')

plt.tight_layout()
plt.show()