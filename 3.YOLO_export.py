import os
import torch
from ultralytics import YOLO
modelname = 'yolov5s6u'
in_shape = (1,3,1280,1280)
fp16=True
enginename = f"{modelname}.{in_shape}{'.FP16'if fp16 else ''}.engine"

dummy_input = torch.rand(in_shape).cuda()
model = YOLO(f"{modelname}.pt")

if not os.path.isfile(enginename):
    model.export(format="engine", batch=in_shape[0], half=fp16, imgsz=in_shape[-1])
    os.rename(f"{modelname}.engine",enginename)

tensorrt_model = YOLO(enginename)
tensorrt_model.MODE(imgsz=in_shape[-1])
results = tensorrt_model('bus.jpg')#"https://ultralytics.com/images/bus.jpg")
print([r.boxes.conf for r in results])