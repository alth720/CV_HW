import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 이미지 불러오기
src = cv.imread('coffee cup.JPG')
src_rgb = cv.cvtColor(src, cv.COLOR_BGR2RGB)

# 초기 사각형 설정 (x, y, width, height)
rect = (100, 90, 1100, 800)

# GrabCut을 위한 초기 마스크 및 모델 생성
mask = np.zeros(src.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# GrabCut 실행 (5회 반복)
cv.grabCut(src, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# 마스크 후처리: 전경(1) 및 전경(3)만 1, 나머지는 0
mask2 = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype('uint8')

# 배경 제거된 이미지 생성
result = src_rgb * mask2[:, :, np.newaxis]

# 마스크 시각화용 이미지 생성
visual_mask = (mask2 * 255).astype('uint8')

# 시각화
plt.figure(figsize=(15, 5))

# 원본 이미지
plt.subplot(1, 3, 1)
plt.imshow(src_rgb)
plt.title("Original Image")
plt.axis('off')

# 마스크 이미지
plt.subplot(1, 3, 2)
plt.imshow(visual_mask, cmap='gray')
plt.title("GrabCut Mask")
plt.axis('off')

# 배경 제거 이미지
plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title("Foreground Only")
plt.axis('off')

plt.tight_layout()
plt.show()
