import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread('IMG/img1.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('IMG/img2.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create(0, 3, 0.1, 10, 2)
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)


bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

# Ratio test 적용
T = 0.7
good_match = []
for nearest1, nearest2 in matches:
    if (nearest1.distance/nearest2.distance) < T:
        good_match.append(nearest1)

print(f"Good matches found: {len(good_match)}")

# 매칭점 추출
points1 = np.float32([kp1[m.queryIdx].pt for m in good_match])
points2 = np.float32([kp2[m.trainIdx].pt for m in good_match])

# 호모그래피 행렬 계산 (RANSAC 사용)
H, _ = cv.findHomography(points1, points2, cv.RANSAC)

# 투시 변환 적용 (img1을 img2에 정렬)
warped_img = cv.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

img_match = cv.drawMatches(img1, kp1, img2, kp2, good_match, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(12, 4))

# 1. 원본 이미지
plt.subplot(2, 2, 1)
plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# 2. 투시 변환 이미지
plt.subplot(2, 2, 2)
plt.imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB))
plt.title("Warped (Homography)")
plt.axis('off')

# 3. 오버레이 이미지
overlay = cv.addWeighted(warped_img, 0.5, img2, 0.5, 0)  # 두 이미지 평균값
plt.subplot(2, 2, 3)
plt.imshow(cv.cvtColor(overlay, cv.COLOR_BGR2RGB))
plt.title("Overlay (Transparent)")
plt.axis('off')

# 4. 특징점 매칭 이미지
plt.subplot(2, 2, 4)
plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
plt.title("Feature Matching")
plt.axis('off')

plt.tight_layout()
plt.show()