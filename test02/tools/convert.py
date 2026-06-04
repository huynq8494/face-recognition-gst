from ultralytics import YOLO

model = YOLO("yolo26n.pt")
success = model.export(format="onnx", dynamic=True)
