import enum
import torch
import tensorrt as trt
# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class Debayer5x5(torch.nn.Module):
    # from https://github.com/cheind/pytorch-debayer
    """Demosaicing of Bayer images using Malver-He-Cutler algorithm.

    Requires BG-Bayer color filter array layout. That is,
    the image[1,1]='B', image[1,2]='G'. This corresponds
    to OpenCV naming conventions.

    Compared to Debayer2x2 this method does not use upsampling.
    Compared to Debayer3x3 the algorithm gives sharper edges and
    less chromatic effects.

    ## References
    Malvar, Henrique S., Li-wei He, and Ross Cutler.
    "High-quality linear interpolation for demosaicing of Bayer-patterned
    color images." 2004
    """
    class Layout(enum.Enum):
        """Possible Bayer color filter array layouts.

        The value of each entry is the color index (R=0,G=1,B=2)
        within a 2x2 Bayer block.
        """
        RGGB = (0, 1, 1, 2)
        GRBG = (1, 0, 2, 1)
        GBRG = (1, 2, 0, 1)
        BGGR = (2, 1, 1, 0)

    def __init__(self, layout: Layout = Layout.RGGB):
        super(Debayer5x5, self).__init__()
        self.layout = layout
        # fmt: off
        self.kernels = torch.nn.Parameter(
            torch.tensor(
                [
                    # G at R,B locations
                    # scaled by 16
                    [ 0,  0, -2,  0,  0], # noqa
                    [ 0,  0,  4,  0,  0], # noqa
                    [-2,  4,  8,  4, -2], # noqa
                    [ 0,  0,  4,  0,  0], # noqa
                    [ 0,  0, -2,  0,  0], # noqa

                    # R,B at G in R rows
                    # scaled by 16
                    [ 0,  0,  1,  0,  0], # noqa
                    [ 0, -2,  0, -2,  0], # noqa
                    [-2,  8, 10,  8, -2], # noqa
                    [ 0, -2,  0, -2,  0], # noqa
                    [ 0,  0,  1,  0,  0], # noqa

                    # R,B at G in B rows
                    # scaled by 16
                    [ 0,  0, -2,  0,  0], # noqa
                    [ 0, -2,  8, -2,  0], # noqa
                    [ 1,  0, 10,  0,  1], # noqa
                    [ 0, -2,  8, -2,  0], # noqa
                    [ 0,  0, -2,  0,  0], # noqa

                    # R at B and B at R
                    # scaled by 16
                    [ 0,  0, -3,  0,  0], # noqa
                    [ 0,  4,  0,  4,  0], # noqa
                    [-3,  0, 12,  0, -3], # noqa
                    [ 0,  4,  0,  4,  0], # noqa
                    [ 0,  0, -3,  0,  0], # noqa

                    # R at R, B at B, G at G
                    # identity kernel not shown
                ]
            ).view(4, 1, 5, 5).float() / 16.0,
            requires_grad=False,
        )
        # fmt: on

        self.index = torch.nn.Parameter(
            # Below, note that index 4 corresponds to identity kernel
            self._index_from_layout(layout),
            requires_grad=False,
        )

    def forward(self, x):
        """Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        """
        B, C, H, W = x.shape

        xpad = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
        planes = torch.nn.functional.conv2d(xpad, self.kernels, stride=1)
        planes = torch.cat(
            (planes, x), 1
        )  # Concat with input to give identity kernel Bx5xHxW
        rgb = torch.gather(
            planes,
            1,
            self.index.repeat(
                1,
                1,
                torch.div(H, 2, rounding_mode="floor"),
                torch.div(W, 2, rounding_mode="floor"),
            ).expand(
                B, -1, -1, -1
            ),  # expand for singleton batch dimension is faster
        )
        return torch.clamp(rgb, 0, 1)

    def _index_from_layout(self, layout: Layout = Layout) -> torch.Tensor:
        """Returns a 1x3x2x2 index tensor for each color RGB in a 2x2 bayer tile.

        Note, the index corresponding to the identity kernel is 4, which will be
        correct after concatenating the convolved output with the input image.
        """
        #       ...
        # ... b g b g ...
        # ... g R G r ...
        # ... b G B g ...
        # ... g r g r ...
        #       ...
        # fmt: off
        rggb = torch.tensor(
            [
                # dest channel r
                [4, 1],  # pixel is R,G1
                [2, 3],  # pixel is G2,B
                # dest channel g
                [0, 4],  # pixel is R,G1
                [4, 0],  # pixel is G2,B
                # dest channel b
                [3, 2],  # pixel is R,G1
                [1, 4],  # pixel is G2,B
            ]
        ).view(1, 3, 2, 2)
        # fmt: on
        return {
            layout.RGGB: rggb,
            layout.GRBG: torch.roll(rggb, 1, -1),
            layout.GBRG: torch.roll(rggb, 1, -2),
            layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
        }.get(layout)

def build_torch_model():
    dummy_input = torch.rand((1,1,224,224)).cuda()
    # Instantiate the Debayer5x5 model
    model = Debayer5x5()
    # Set the model to evaluation mode
    model.eval().cuda()

    print(model(dummy_input).shape)

    return model,dummy_input

def export_torch_onnx_model(model,x,onnx_model_path):
    # Export to ONNX
    torch.onnx.export(
        model,
        x,
        onnx_model_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"]
    )
    print(f"ONNX model saved to {onnx_model_path}")

def build_static_engine(onnx_file_path, engine_file_path, fp16=True):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:
        
        # Enable FP16 mode if supported
        if fp16 and builder.platform_has_fast_fp16:
            print("Platform supports FP16, enabling FP16 optimization...")
            config.set_flag(trt.BuilderFlag.FP16)

        # Parse the ONNX file
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Build the engine
        engine = builder.build_serialized_network(network, builder.create_builder_config())
        if engine is None:
            print("Failed to build the engine.")
            return None

        # Save the engine to a file
        with open(engine_file_path, "wb") as f:
            f.write(engine)
        print(f"Static TensorRT engine saved to {engine_file_path}")


modelname = "debayer5x5"
model,x = build_torch_model()
torch.save(model.state_dict(), f'{modelname}.pth')

torch_file_path = f'{modelname}.pth'
onnx_file_path = f'{modelname}.onnx'
engine_file_path = f'{modelname}.trt'

export_torch_onnx_model(model,x,onnx_file_path)
build_static_engine(onnx_file_path, engine_file_path)
