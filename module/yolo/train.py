from ultralytics import YOLO

# Load a COCO-pretrained YOLO26n model
model = YOLO("yolo5n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="data.yaml", epochs=100, imgsz=640)

# # Run inference with the YOLO26n model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")