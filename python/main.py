from ultralytics import YOLO
import cv2 as cv

model = YOLO("yolov8n.pt")

cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    results = model(frame)

    box_frame = results[0].plot()

    cv.imshow("Tugas GMRT Day 2", box_frame)
    if cv.waitKey(1) == ord('q'):
        break

cam.release()
cv.destroyAllWindows()