from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
results = model.train(data="data.yaml", epochs=100, imgsz=640, batch=32)