import cv2
import numpy as np
import onnxruntime as ort
import onnx
import ast

onnx_model = onnx.load("yolo26n.onnx")
props = {p.key: p.value for p in onnx_model.metadata_props}
names = ast.literal_eval(props['names']) if 'names' in props else {}

session = ort.InferenceSession("yolo26n.onnx", providers=['CPUExecutionProvider'])
img = cv2.imread("/home/huynq/data/images/baggage_claim.jpg")
h, w = img.shape[:2]

input_data = cv2.resize(img, (640, 640)).transpose(2, 0, 1)
input_data = input_data[np.newaxis, :, :, :].astype(np.float32) / 255.0

outputs = session.run(None, {session.get_inputs()[0].name: input_data})
predictions = np.squeeze(outputs[0])

for obj in predictions:
    x1, y1, x2, y2, conf, cls_id = obj

    if conf > 0.4:
        x1, x2 = int(x1 * w / 640), int(x2 * w / 640)
        y1, y2 = int(y1 * h / 640), int(y2 * h / 640)
        label = names.get(int(cls_id), f"ID:{int(cls_id)}")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("YOLO26", img)
cv2.waitKey(0)
