import cv2
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator


model = YOLO('yolov8n.pt')
x_line = 100

img = cv2.imread('zidane.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = model.predict(img, conf=0.5, classes=0)
annotator = Annotator(img)

for r in results:
    for box in r.boxes:
        b = box.xyxy[0]
        if b[1] > x_line:
            c = box.cls
            annotator.box_label(b, f"{r.names[int(c)]} {float(box.conf):.2}")

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.line(img, (0, x_line), (img.shape[1] - 1, x_line), (255, 0, 0), 2)

cv2.imshow("YOLO", img)
cv2.waitKey(0)

cv2.destroyAllWindows()
