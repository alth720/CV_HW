import cv2
import numpy as np
from sort import Sort
import time

# 클래스 이름 불러오기
with open("model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# YOLOv4 설정
net = cv2.dnn.readNet("model/yolov4.weights", "model/yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# SORT 초기화
tracker = Sort()

# ✅ 비디오 파일 경로로 변경
video_path = "HW_8/slow_traffic_small.mp4"  # 또는 절대경로: "c:/python_CV/HW_8/slow_traffic_small.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width = frame.shape[:2]

    # YOLO 입력 포맷 설정
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # YOLO 결과 파싱
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    dets = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            dets.append([x, y, x + w, y + h, confidences[i]])

    dets = np.array(dets)

    # SORT 추적기 업데이트
    tracked_objects = tracker.update(dets)

    # 추적된 객체 표시
    for *bbox, obj_id in tracked_objects:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {int(obj_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLOv4 + SORT Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
