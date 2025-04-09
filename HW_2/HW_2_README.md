# CV_HW_2

## 📂 과제 1 – 이진화 및 히스토그램 구하기

### 📌 주요 코드
```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 그레이스케일로 변환
t, bin_img = cv.threshold(gray, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # 이진화, t는 오츄 알고리즘이 찾은 최적 임곗값
h1 = cv.calcHist([gray], [0], None, [256], [0, 256])  # 그레이스케일 히스토그램 계산
h2 = cv.calcHist([bin_img], [0], None, [256], [0, 256])  # 이진화된 이미지 히스토그램 계산
cv.imshow('Binary Image', bin_img)  # 이진화된 이미지 출력
plt.plot(h1, color='b', linewidth=1), plt.show()  # 그레이스케일 히스토그램은 파란색 선으로 출력
plt.plot(h2, color='r', linewidth=1), plt.show()  # 이진화된 이미지의 히스토그램은 빨간색 선으로 출력
```

### ✅ 구현 결과
<img width="1167" alt="image" src="https://github.com/user-attachments/assets/77ca9389-61ea-4d66-8676-67380c0cd9d5" />
<img width="482" alt="image" src="https://github.com/user-attachments/assets/9113423e-22fe-46cc-b8bb-f96d186bf226" />



## 📂 과제 2 – 모폴로지 연산 적용하기

### 📌 주요 코드
```python
b = bin_img[bin_img.shape[0]//2:bin_img.shape[0], 0:bin_img.shape[0]//2+1]  # 이미지 하단 왼쪽 부분 추출
se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # 사각형 커널 생성(5x5)
dilation = cv.morphologyEx(b, cv.MORPH_DILATE, se)  # 팽창(흰색 영역 확장) -> 객체가 더 굵어짐 
erosion = cv.morphologyEx(b, cv.MORPH_ERODE, se)  # 침식(흰색 영역을 줄임) -> 객체가 더 얇아짐
opening = cv.morphologyEx(b, cv.MORPH_OPEN, se)  # 열림(침식 후 팽창 수행) -> 작은 노이즈 제거
closing = cv.morphologyEx(b, cv.MORPH_CLOSE, se)  # 닫힘(팽창 후 침식 수행) -> 내부 구멍 채움
mp = np.hstack((b, dilation, erosion, opening, closing))  # 이어붙여 한 화면에 출력

```

### ✅ 구현 결과
<img width="990" alt="image" src="https://github.com/user-attachments/assets/8f926a30-1b3c-4273-80e7-68718d42f7dd" />
+ 원본  팽창  침식  열림  닫힘  


## 📂 과제 3 – 기하연산및선형보간적용하기

### 📌 주요 코드
```python
rows, cols = img.shape[:2]  # 이미지 크기
center = (cols // 2, rows // 2)  # 중심 좌표
angle = 45  # 회전 각도
scale = 1.5  # 스케일 1.5배
rot_matrix = cv.getRotationMatrix2D(center, angle, scale)  # 회전 변환 행렬 생성
rot_img = cv.warpAffine(img, rot_matrix, (cols, rows), flags=cv.INTER_LINEAR)  # 회전 및 확대, cv.INTER_LINEAR를 사용해 선형 보간 
img_small = cv.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)  # 출력 이미지가 너무 커서 절반으로 줄임
rotImg_small = cv.resize(rot_img, dsize=(0, 0), fx=0.5, fy=0.5)  # 회전 및 확대한 이미지도 마찬가지
imgs = np.hstack((img_small, rotImg_small))  # 이어붙여 한 화면에 출력
```

### ✅ 구현 결과
<img width="895" alt="image" src="https://github.com/user-attachments/assets/388c90b3-c899-485b-bf3a-d2a5f0146b3f" />
