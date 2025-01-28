from ultralytics import YOLO
model = YOLO("yolov5s6u.pt")
model.export(format="engine")#, half=True)
tensorrt_model = YOLO("yolov5s6u.engine")
results = tensorrt_model("https://ultralytics.com/images/bus.jpg")
print(results)