import time
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from GeneralTensorRTModel import GeneralTensorRTInferenceModel
from ultralytics import YOLO

debayer = GeneralTensorRTInferenceModel("debayer5x5.trt")
yolo = YOLO("yolov5s6u.engine")

x_torch_cuda = torch.rand([*debayer.input_shape]).cuda()
print(debayer(x_torch_cuda).shape)
print(yolo(debayer(x_torch_cuda)))
