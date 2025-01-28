import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

try:
    import torch
    numpy_to_torch_dtype_dict = {
                bool: torch.bool,
                np.uint8: torch.uint8,
                np.int8: torch.int8,
                np.int16: torch.int16,
                np.int32: torch.int32,
                np.int64: torch.int64,
                np.float16: torch.float16,
                np.float32: torch.float32,
                np.float64: torch.float64,
                np.complex64: torch.complex64,
                np.complex128: torch.complex128,
            }
except Exception as e:
    print('[warning]: no torch support')
    numpy_to_torch_dtype_dict = None

class GeneralTensorRTInferenceModel:

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __repr__(self):
            return f"HostDeviceMem(host={self.host}, device={self.device})"

    def __init__(self, engine_path, input_tensor_name='input', output_tensor_name='output'):
        # Initialize logger, runtime, and engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine_path = engine_path
        self.engine = self._load_engine()

        # Allocate buffers and create execution context
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        self.context = self.engine.create_execution_context()        
        
        self.output_tensor_name = output_tensor_name        
        self.output_shape = self.engine.get_tensor_shape(self.output_tensor_name)

        self.input_tensor_name = input_tensor_name        
        self.input_shape = self.engine.get_tensor_shape(self.input_tensor_name)

    def _load_engine(self):
        """Load and deserialize a TensorRT engine from a file."""
        with open(self.engine_path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        """Allocate memory buffers for all inputs and outputs."""
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        num_io = self.engine.num_io_tensors  
        for i in range(num_io):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(tensor_name)
            self.dtype = dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            size = trt.volume(shape)
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream
    
    def _transfer_torch_cuda_input(self, model_input, input_mem):
        cuda.memcpy_dtod_async(
            input_mem.device,
            model_input.data_ptr(),
            model_input.element_size() * model_input.nelement(),
            self.stream)

    def _transfer_np_input(self, model_input, input_mem):
        np.copyto(input_mem.host, model_input.ravel())
        cuda.memcpy_htod_async(input_mem.device, input_mem.host, self.stream)

    def _transfer_np_output(self, output_mem, batch_size):
        cuda.memcpy_dtoh_async(output_mem.host, output_mem.device, self.stream)
        return output_mem.host.reshape(batch_size, *self.output_shape[1:])
        
    def _transfer_torch_cuda_output(self, output_mem, batch_size, device=None):
        # Create a PyTorch tensor with the appropriate device
        output_tensor = torch.empty((batch_size, *self.output_shape[1:]), device=device,
                                    dtype=numpy_to_torch_dtype_dict[self.dtype])
        # Perform device-to-device memory copy
        cuda.memcpy_dtod_async(
            int(output_tensor.data_ptr()),  # Destination (PyTorch CUDA tensor pointer)
            int(output_mem.device),        # Source (output_mem.device pointer)
            output_mem.host.nbytes,        # Number of bytes to copy
            self.stream                     # CUDA stream for asynchronous operation
        )
        return output_tensor
            
    def __call__(self,input_data):
        return self.infer([input_data])[0]

    def infer(self, model_inputs):
        is_numpy = isinstance(model_inputs[0], np.ndarray)
        if numpy_to_torch_dtype_dict:
            is_torch = torch.is_tensor(model_inputs[0])
        else:
            is_torch = False
        assert is_numpy or is_torch, "Unsupported input data format!"
        
        # Check batch size consistency
        batch_sizes = [x.shape[0] if is_numpy else x.size(0) for x in model_inputs]
        assert len(set(batch_sizes)) == 1, "Input batch sizes are inconsistent!"
        batch_size = batch_sizes[0]

        if is_torch:
            xt:torch.Tensor = model_inputs[0]
            assert xt.is_cuda, "Unsupported input torch on cpu!"
            assert xt.dtype == numpy_to_torch_dtype_dict[self.dtype], f"dtype are inconsistent, need {self.dtype}"
            transfer_input = self._transfer_torch_cuda_input
            transfer_output = lambda output_mem, batch_size:self._transfer_torch_cuda_output(
                                                            output_mem, batch_size,xt.device)

        elif is_numpy:
            x:np.ndarray = model_inputs[0]
            assert x.dtype == self.dtype, f"dtype are inconsistent, need {self.dtype}"
            transfer_input = self._transfer_np_input
            transfer_output = self._transfer_np_output
        
        for i, model_input in enumerate(model_inputs):            
            transfer_input(model_input, self.inputs[i])
        self.stream.synchronize()
        
        # Run inference
        self.context.execute_v2(bindings=self.bindings)

        # Transfer output data to host
        outputs = [transfer_output(out, batch_size) for out in self.outputs]
        self.stream.synchronize()
        return outputs