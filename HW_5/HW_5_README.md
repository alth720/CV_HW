# CV_HW_3

## 📂 과제 1 – 소벨 에지 검출 및 결과 시각화

### 📌 주요 코드
```python
grad_x = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=3)  # x 방향 소벨
grad_y = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=3)  # y 방향 소벨
magnitude = cv.magnitude(grad_x, grad_y)  # 엣지 강도 계산
edge_strength = cv.convertScaleAbs(magnitude)  # 절댓값을 취해 양수 영상으로 변환
plt.figure(figsize=(10, 5))  # 시각화
```

### ✅ 구현 결과
<img width="745" alt="image" src="https://github.com/user-attachments/assets/9b0e80e0-1466-4df6-9dcc-9c8ee30e6ada" />



## 📂 과제 2 – 캐니 에지 및 허프 변환을 이용한 직선 검출

### 📌 주요 코드
```python
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

plt.figure(figsize=(10, 5))  # 시각화
```

### ✅ 구현 결과
<img width="748" alt="image" src="https://github.com/user-attachments/assets/c20f6c56-a4a4-4b84-8d5b-657cba0a996b" />  


## 📂 과제 3 – GrabCut을 이용한 대화식 영역 분할 및 객체 추출

### 📌 주요 코드
```python
rect = (100, 90, 1100, 800)  # 초기 사각형 설정(x, y, width, height)

# GrabCut을 위한 초기 마스크 및 모델 생성
mask = np.zeros(src.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64) 

cv.grabCut(src, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)  # #GrabCut 실행 (5회 반복)

mask2 = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype('uint8')  # 마스크 후처리: 전경(1) 및 전경(3)만 1, 나머지는 0
result = src_rgb * mask2[:, :, np.newaxis]  # 배경 제거된 결과 이미지 생성
visual_mask = (mask2 * 255).astype('uint8')  # 마스크 시각화용 이미지 생성
```

### ✅ 구현 결과
<img width="1121" alt="image" src="https://github.com/user-attachments/assets/d35011ce-0a70-4e1e-9cf0-f29e6f239afd" />
