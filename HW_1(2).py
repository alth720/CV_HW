import numpy as np
import cv2 as cv
import sys

<<<<<<< HEAD
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
=======
cap = cv.VideoCapture(0, cv.CAP_DSHOW) # 카메라와 연결 시도 
>>>>>>> 6b14898 (+HW2)

if not cap.isOpened():
    sys.exit('카메라 연결 실패')

while True:
<<<<<<< HEAD
    ret, frame = cap.read()
=======
    ret, frame = cap.read() # 비디오를 구성하는 프레임 획득 
>>>>>>> 6b14898 (+HW2)
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    canny = cv.Canny(gray, 100, 200)

    gray_cap=cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
    
    caps=np.hstack((frame,gray_cap))

    cv.imshow('Video display',caps)

<<<<<<< HEAD
    key = cv.waitKey(1)
=======
    key = cv.waitKey(1) # 1밀리초 동안 키보드 입력 기다림
>>>>>>> 6b14898 (+HW2)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
