# CV_HW_4

## 📂 과제 1 –  SIFT를 이용한 특징점 검출 및 시각화

### 📌 주요 코드
```python
sift = cv.SIFT_create(0, 3, 0.1, 10, 2)  # SIFT 객체 생성

cv2.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
# nfeatures: 검출 최대 특징 수
# nOctaveLayers: 이미지 피라미드에 사용할 계층 수
# contrastThreshold: 필터링할 빈약한 특징 문턱 값
# edgeThreshold: 필터링할 엣지 문턱 값
# sigma: 이미지 피라미드 0 계층에서 사용할 가우시안 필터의 시그마 값

kp, des = sift.detectAndCompute(gray, None)  # 키 포인트 검출 및 서술자 계산

gray = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 특징점 시각화
# cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS 플래그에 의해서 원의 크기는 특징점이 검출된 스케일 크기의 영향을 받으며, 검출된 스케일에 비례하는 크기의 원이 그려짐(원이 클수록 해당 특징점은 더 큰 영역에서 의미 있는 패턴을 가지고 있으며, 작은 원은 더 세밀한 특징을 나타낸다고 해석할 수 있음)

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # 원본 이미지 표시(matplotlib는 BGR이 아니라 RGB)
plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))  # 특징점 이미지 표시
```

### ✅ 구현 결과
<img width="570" alt="image" src="https://github.com/user-attachments/assets/d23aaaa8-158c-4a67-9b5c-67d77e858439" />



## 📂 과제 2 –  SIFT를 이용한 두 영상 간 특징점 매칭

### 📌 주요 코드
```python
img1 = cv.imread('mot_color70.jpg')[190:350, 440:560]  # 버스를 크롭하여 모델 영상으로 사용
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)  # 버스를 크롭하여 모델 영상으로 사용
img2 = cv.imread('mot_color83.jpg')  # img2는 원본 크기 그대로 사용
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)  # img2 그레이 스케일로 변환

sift = cv.SIFT_create(0, 3, 0.1, 10, 2)  # SIFT 객체 생성

# gray1, gray2 키 포인트 검출 및 서술자 계산
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# FLANN 매칭 설정
FLANN_INDEX_KDTREE = 1  # 인덱스 파라미터 설정
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
search_params = dict(checks=90)  # 검색 파라미터 설정

matcher = cv.FlannBasedMatcher(index_params, search_params)  # Flann 매처 생성
matches = matcher.knnMatch(des1, des2, k=2)  # 매칭 계산

# Ratio Test 적용
T = 0.7
good_match = []
for nearest1, nearest2 in matches:
    if (nearest1.distance / nearest2.distance) < T:
        good_match.append(nearest1)

img_match = cv.drawMatches(img1, kp1, img2, kp2, good_match, None,
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # 매칭 그리기
plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))  # 특징점 매칭 이미지

# 또는 BFMatcher 사용:
# bf = cv.BFMatcher(cv.NORM_L2)
```

### ✅ 구현 결과
<img width="551" alt="image" src="https://github.com/user-attachments/assets/c1d6a849-8552-46b0-85d3-0e239a8666e4" />  


## 📂 과제 3 – 호모그래피를 이용한 이미지 정합(Image Alignment)

### 📌 주요 코드
```python
bf = cv.BFMatcher(cv.NORM_L2)  # Brute Force 매칭기 생성
# cv2.BFMatcher_create(normType, crossCheck)
# normType: 거리 측정 알고리즘 (cv2.NORM_L1, cv2.NORM_L2(default), cv2.NORM_L2SQR, cv2.NORM_HAMMING, cv2.NORM_HAMMING2)
# crosscheck: 상호 매칭이 되는 것만 반영 (default=False)

matches = bf.knnMatch(des1, des2, k=2)  # 매칭 계산

# 매칭된 점 추출
points1 = np.float32([kp1[m.queryIdx].pt for m in good_match])
points2 = np.float32([kp2[m.trainIdx].pt for m in good_match]) 

H, _ = cv.findHomography(points1, points2, cv.RANSAC)  # 호모그래피 행렬 계산 (RANSAC 사용)

warped_img = cv.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))  # 투시 변환 적용 (img1을 img2에 정렬)
overlay = cv.addWeighted(warped_img, 0.5, img2, 0.5, 0)  # 두 이미지 합성 (반투명)

```

### ✅ 구현 결과
<img width="896" alt="image" src="https://github.com/user-attachments/assets/542836fe-a009-48c2-bf31-53fa079d5229" />
