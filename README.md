<<<<<<< HEAD
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
=======
# CV_HW
CVclass_HW_SMILE
>>>>>>> 6b14898 (+HW2)
