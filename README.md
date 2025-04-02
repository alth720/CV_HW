# CV_HW_1
CVclass_HW_SMILE

# 과제 1

+ 주요 코드
 
  + gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 이미지를 회색으로 바꿈.
  + gray_img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # 원본 이미지와 같이 3차원으로 만들어줌.
  + imgs = np.hstack((img,gray_img)) # imgs에 원본 이미지와, 회색으로 바꾼 이미지를 가로로 이어붙임.


 + 구현 결과

   + <img width="1210" alt="image" src="https://github.com/user-attachments/assets/34de0cfc-c2fb-4d38-a2c5-4798fc39aede" />


# 과제 2

+ 주요 코드
 
  + cap = cv.VideoCapture(0, cv.CAP_DSHOW) # 카메라와 연결 시도
  + ret, frame = cap.read() # 비디오를 구성하는 프레임 획득
  + gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # 프레임을 회색으로 바꿈.
  + canny = cv.Canny(gray, 100, 200) # 엣지 검출. 100: 하단 임계값, 200: 상단 임계값. 임계값이 커질수록 검출되는 엣지가 적어진다.
  + caps=np.hstack((frame,gray_cap)) # 원본 비디오와 엣지 검출 비디오를 이어 붙임.
  + if key == ord('q'):
        break                       # 'q'를 누르면 종료.


 + 구현 결과

   + <img width="962" alt="image" src="https://github.com/user-attachments/assets/918f3e00-3073-4ca9-a30f-2113097d19cd" />



# 과제 3

+ 주요 코드
 
  + clone = img.copy() # 원본 이미지 복사 (초기화 용도)
  + roi = None  #관심영역 저장 변수
  + drawing = False  # 드래그 상태
  + ix, iy = -1, -1  # 시작점 좌표
  + global ix, iy, drawing, roi, img # 여러 번의 함수 호출에서도 값을 유지하기 위해 전역 변수로 선언.
  + if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y                # 왼쪽 마우스 버튼이 내려갔을 때 좌표 저장. 드래그 상태를 True.
  + elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            Image = img.copy()
            cv.rectangle(Image, (ix, iy), (x, y), (0, 0, 255), 2)
            cv.imshow('Image', Image)         # 드래그 상태 Treu이고 마우스가 움직일 때, 원본 이미지 복사본 Image에 빨간색 직사각형 시각화.
  + elif event == cv.EVENT_LBUTTONUP:         # 왼쪽 버튼이 올라올 때
        drawing = False                       # 드래그 상태: False
        cv.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 2)
        roi = clone[iy:y, ix:x]               #관심영역 추출, [높이, 너비] (ix, iy) -> (x, y)까지의 영역을 잘라냄. 

        if roi.size > 0:                      # 빈 영역 방지
            cv.imshow('ROI', roi)
  + if key == ord('q'):
        break                         # 'q'키 누르면 종료.
  + elif key == ord('r'):
        img = clone.copy()
        cv.destroyWindow('ROI')        # 'r'키 누르면 리셋. 원본 이미지를 복사해둔 clone으로 복귀.
  + elif key == ord('s') and roi is not None: 
        cv.imwrite('selected_roi.jpg', roi)
        print('선택한 영역이 저장되었습니다: selected_roi.jpg')   # 's'키 누르면 관심영역이 있을 때 'selected_roi.jpg'파일로 저장됨(cv.imwrite함수 사용). 

     


 + 구현 결과

   + <img width="648" alt="image" src="https://github.com/user-attachments/assets/e52440d4-c03f-4f7d-81c1-26191c186db5" />

# CV_HW_2
CVclass_HW_SMILE

# 과제 1

+ 주요 코드
 
  + gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 그레이스케일로 변환.
  + t, bin_img = cv.threshold(gray, 127, 255, cv.THRESH_BINARY+cv.THRESH_OTSU) # 이진화, t는 오츄 알고리즘이 찾은 최적 임곗값
  + h1 = cv.calcHist([gray], [0], None, [256], [0, 256]) # 그레이스케일 히스토그램 계산
  + h2 = cv.calcHist([bin_img], [0], None, [256], [0, 256]) # 이진화된 이미지 히스토그램 계산
  + cv.imshow('Binary Image', bin_img) # 이진화된 이미지 출력.
  + plt.plot(h1, color = 'b', linewidth = 1), plt.show() # 그레이스케일 히스토그램은 파란색 선으로 출력.
  + plt.plot(h2, color = 'r', linewidth = 1), plt.show() # 이진화된 이미지의 히스토그램은 빨간색 선으로 출력. 


 + 구현 결과

   + <img width="1167" alt="image" src="https://github.com/user-attachments/assets/77ca9389-61ea-4d66-8676-67380c0cd9d5" />
   <img width="482" alt="image" src="https://github.com/user-attachments/assets/9113423e-22fe-46cc-b8bb-f96d186bf226" />



# 과제 2

+ 주요 코드
 
  + b = bin_img[bin_img.shape[0]//2:bin_img.shape[0], 0:bin_img.shape[0]//2+1] # 이미지의 특정 부분을 잘라냄.
  + se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5)) # 사각형 커널(5x5) 생성.
  + dilation = cv.morphologyEx(b, cv.MORPH_DILATE, se) # 팽창(흰색 영역 확장) -> 객체가 더 굵어짐. 
  + erosion = cv.morphologyEx(b, cv.MORPH_ERODE, se) # 침식(흰색 영역을 줄임) -> 객체가 더 얇아짐. 
  + opening = cv.morphologyEx(b, cv.MORPH_OPEN, se) # 열림(침식 후 팽창 수행) -> 작은 노이즈 제거. 
  + closing = cv.morphologyEx(b, cv.MORPH_CLOSE, se) # 닫힘(팽창 후 침식 수행) -> 내부 구멍 채움.
  + mp = np.hstack((b, dilation, erosion, opening, closing)) # 이어붙여 한 화면에 출력하도록 함. 

 + 구현 결과

   + <img width="990" alt="image" src="https://github.com/user-attachments/assets/8f926a30-1b3c-4273-80e7-68718d42f7dd" />
   +           원본                     팽창                    침식                   열림                    닫힘 


# 과제 3

+ 주요 코드
 
  + rows, cols = img.shape[:2]  # 이미지의 크기
  + center = (cols // 2, rows // 2)  # 중심 좌표
  + angle = 45  # 회전 각도
  + scale = 1.5  # 1.5배 확대
  + rot_matrix = cv.getRotationMatrix2D(center, angle, scale) # 회전 변환 행렬 생성
  + rot_img = cv.warpAffine(img, rot_matrix, (cols, rows), flags=cv.INTER_LINEAR) # 회전 및 확대, cv.INTER_LINEAR를 사용해 선형 보
  + img_small = cv.resize(img, dsize = (0, 0), fx = 0.5, fy = 0.5) # 출력 이미지가 너무 커서 절반으로 줄임.
  + rotImg_small = cv.resize(rot_img, dsize = (0, 0), fx = 0.5, fy = 0.5) # 회전 및 확대한 이미지도 마찬가지.
  + imgs = np.hstack((img_small, rotImg_small)) # 이어붙여 한 화면에 출력할 수 있도록 함. 

 + 구현 결과

   + <img width="895" alt="image" src="https://github.com/user-attachments/assets/388c90b3-c899-485b-bf3a-d2a5f0146b3f" />


# CV_HW_3
CVclass_HW_SMILE

# 과제 1

+ 주요 코드
 
  + grad_x = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize = 3) # 소벨 연산자 적용
  + grad_y = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize = 3)
  + magnitude = cv.magnitude(grad_x, grad_y) # 에지 강도 계산
  + edge_strength = cv.convertScaleAbs(magnitude) # 절댓값을 취해 양수 영상으로 변환
  + plt.figure(figsize=(10, 5)) # matplotlib로 시각화


 + 구현 결과

   + <img width="745" alt="image" src="https://github.com/user-attachments/assets/9b0e80e0-1466-4df6-9dcc-9c8ee30e6ada" />



# 과제 2

+ 주요 코드
 
  + lines = cv.HoughLinesP(
        canny,              # 입력 에지 이미지
        rho=1,              # 거리 해상도 (픽셀 단위)
        theta=np.pi/180,    # 각도 해상도 (라디안 단위)
        threshold=60,       # 직선으로 판단할 최소 교차점 수
        minLineLength=40,   # 직선으로 간주할 최소 길이
        maxLineGap=20       # 동일 선상의 점들을 연결할 최대 간격
    )  # 허프 변환을 통한 직선 검출

  + if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 직선 그리기 
  + plt.figure(figsize=(10, 5))  # matplotlib로 원본 이미지 + 직선 검출 이미지 나란히 표시

 + 구현 결과

   + <img width="748" alt="image" src="https://github.com/user-attachments/assets/c20f6c56-a4a4-4b84-8d5b-657cba0a996b" />




# 과제 3

+ 주요 코드
 
  + rect = (100, 90, 1100, 800)  # 초기 사각형 설정 (x, y, width, height)
  + mask = np.zeros(src.shape[:2], np.uint8)
  + bgdModel = np.zeros((1, 65), np.float64)
  + fgdModel = np.zeros((1, 65), np.float64)  # GrabCut을 위한 초기 마스크 및 모델 생성
  + cv.grabCut(src, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)  #GrabCut 실행 (5회 반복)
  + mask2 = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype('uint8')  # 마스크 후처리: 전경(1) 및 전경(3)만 1, 나머지는 0
  + result = src_rgb * mask2[:, :, np.newaxis]  # 배경 제거된 이미지 생성
  + visual_mask = (mask2 * 255).astype('uint8')  # 마스크 시각화용 이미지 생성


 + 구현 결과

   + <img width="1121" alt="image" src="https://github.com/user-attachments/assets/d35011ce-0a70-4e1e-9cf0-f29e6f239afd" />


# CV_HW_4
CVclass_HW_SMILE

# 과제 1( SIFT를 이용한특징점검출및시각화)

+ 주요 코드
 
  + sift = cv.SIFT_create(0, 3, 0.1, 10, 2) # SIFT 객체 생성. cv2..SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
nfeatures: 검출 최대 특징 수
nOctaveLayers: 이미지 피라미드에 사용할 계층 수
contrastThreshold: 필터링할 빈약한 특징 문턱 값
edgeThreshold: 필터링할 엣지 문턱 값
sigma: 이미지 피라미드 0 계층에서 사용할 가우시안 필터의 시그마 값
  + kp, des = sift.detectAndCompute(gray, None)  # 키 포인트 검출 및 서술자 계산
  + gray = cv.drawKeypoints(gray, kp, None, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 키포인트 그리기 **cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS 플래그에 의해서 원의 크기는 특징점이 검출된 스케일 크기의 영향을 받으며, 검출된 스케일에 비례하는 크기의 원이 그려짐(원이 클수록 해당 특징점은 더 큰 영역에서 의미 있는 패턴을 가지고 있으며, 작은 원은 더 세밀한 특징을 나타낸다고 해석할 수 있음)**
  + plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) # matplotlib는 BGR이 아니라 RGB
  + plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))


 + 구현 결과

   + <img width="570" alt="image" src="https://github.com/user-attachments/assets/d23aaaa8-158c-4a67-9b5c-67d77e858439" />




# 과제 2

+ 주요 코드
 
  + img1 = cv.imread('mot_color70.jpg')[190:350, 440:560]  # 버스를 크롭하여 모델 영상으로 사용
  + gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) # img1 그레이 스케일로 변환
  + img2 = cv.imread('mot_color83.jpg')  # img2는 원본 크기 그대로 사용
  + gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)  # img2 그레이 스케일로 변환
  + sift = cv.SIFT_create(0, 3, 0.1, 10, 2)  # SIFT 객체 생성
  + kp1, des1 = sift.detectAndCompute(gray1, None)
  + kp2, des2 = sift.detectAndCompute(gray2, None)  # gray1, gray2 키 포인트 검출 및 서술자 계산
  + FLANN_INDEX_KDTREE = 1  # 인덱스 파라미터 설정
  + index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
  + search_params = dict(checks=90) # 검색 파라미터 설정
  + matcher = cv.FlannBasedMatcher(index_params, search_params)  # Flann 매처 생성
  + matches = matcher.knnMatch(des1, des2, k = 2)  # 매칭 계산
  + T = 0.7
    good_match = []
    for nearest1, nearest2 in matches:
        if (nearest1.distance/nearest2.distance) < T:
            good_match.append(nearest1)  # Ratio Test 적용
  + img_match = cv.drawMatches(img1, kp1, img2, kp2, good_match, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) # 매칭 그리기
  + plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))  # 특징점 매칭 이미지
 
  + BFMatcher 생성(L2 norm은 SIFT에 적합) 하고 싶을 땐
  + bf = cv.BFMatcher(cv.NORM_L2)  # 이 코드로 간단하게 생성 가능 


 + 구현 결과

   + <img width="551" alt="image" src="https://github.com/user-attachments/assets/c1d6a849-8552-46b0-85d3-0e239a8666e4" />





# 과제 3

+ 주요 코드
 
  + bf = cv.BFMatcher(cv.NORM_L2)  # cv2.BFMatcher_create(normType, crossCheck) **normType: 거리 측정 알고리즘 (cv2.NORM_L1, cv2.NORM_L2(default), cv2.NORM_L2SQR, cv2.NORM_HAMMING, cv2.NORM_HAMMING2)
crosscheck: 상호 매칭이 되는 것만 반영 (default=False)**
  + matches = bf.knnMatch(des1, des2, k=2)  # 매칭 계산
  + points1 = np.float32([kp1[m.queryIdx].pt for m in good_match])
  + points2 = np.float32([kp2[m.trainIdx].pt for m in good_match])  # 매칭점 추출
  + H, _ = cv.findHomography(points1, points2, cv.RANSAC)  # 호모그래피 행렬 계산 (RANSAC 사용)
  + warped_img = cv.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))  # 투시 변환 적용 (img1을 img2에 정렬)
  + overlay = cv.addWeighted(warped_img, 0.5, img2, 0.5, 0)  # 두 이미지(원본 이미지, 투시 변환 적용 이미지)를 반투명하게 변환.


 + 구현 결과

   + <img width="896" alt="image" src="https://github.com/user-attachments/assets/542836fe-a009-48c2-bf31-53fa079d5229" />





