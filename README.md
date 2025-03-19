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
