import numpy as np
import cv2 as cv
import sys

img = cv.imread('IMG/soccer.jpg')
imgs_small = cv.resize(img, dsize = (0, 0), fx = 0.5, fy = 0.5)

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

clone = imgs_small.copy() #원본 이미지 복사 (초기화 용도)
roi = None  #관심영역 저장 변수 

drawing = False  # 드래그 상태
ix, iy = -1, -1  # 시작점 좌표

def draw(event, x, y, flags, param):
    global ix, iy, drawing, roi, img

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            Image = imgs_small.copy()
            cv.rectangle(Image, (ix, iy), (x, y), (0, 0, 255), 2)
            cv.imshow('Image', Image)

    elif event == cv.EVENT_LBUTTONUP:  
        drawing = False
        cv.rectangle(imgs_small, (ix, iy), (x, y), (0, 0, 255), 2)
        roi = clone[iy:y, ix:x]  #관심영역 추출

        if roi.size > 0:  # 빈 영역 방지
            cv.imshow('ROI', roi)


cv.namedWindow('Image')
cv.setMouseCallback('Image', draw)

while True:
    cv.imshow('Image', imgs_small)
    key = cv.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('r'):
        imgs_small = clone.copy()
        cv.destroyWindow('ROI')
    elif key == ord('s') and roi is not None: 
        cv.imwrite('selected_roi.jpg', roi)
        print('선택한 영역이 저장되었습니다: selected_roi.jpg')

cv.destroyAllWindows()
