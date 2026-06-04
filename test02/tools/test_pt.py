import cv2
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
results = model("/home/huynq/data/images/baggage_claim.jpg")
annotated_frame = results[0].plot()
cv2.imshow("YOLO26 Object Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()