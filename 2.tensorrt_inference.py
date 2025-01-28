import time
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from GeneralTensorRTModel import GeneralTensorRTInferenceModel

engine_file_path = "debayer5x5.trt"
trt_infer = GeneralTensorRTInferenceModel(engine_file_path)
x_torch_cuda = torch.rand([*trt_infer.input_shape]).cuda()
x_np = x_torch_cuda.cpu().numpy()

print(trt_infer.predict(x_torch_cuda).shape)
print(trt_infer.predict(x_np).shape)

# will raise error
# print(trt_infer.predict(x_np.tolist()))
# print(trt_infer.predict(x_torch_cuda.cpu()))
