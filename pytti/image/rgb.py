from PIL import Image
import torch as th
from torch.nn import functional as F
from torchvision.transforms import functional as T
from pytti.image import DifferentiableImage
from neurpy.util import clamp_with_grad

class RGBImage(DifferentiableImage):
    def __init__(self, width, height, scale=1, device='cuda'):
        super().__init__(width * scale, height * scale)
        self.tensor = th.nn.Parameter(th.zeros(1, 3, height, width).to(device=device, memory_format=th.channels_last))
        self.output_axes = ('n', 's', 'y', 'x')
        self.scale = scale

    def decode_tensor(self):
        width, height = self.image_shape
        out = F.interpolate(self.tensor, (height, width) , mode='nearest')
        return clamp_with_grad(out,0,1)

    def clone(self):
        width, height = self.image_shape
        dummy = RGBImage(width//self.scale, height//self.scale, self.scale)
        with th.no_grad():
            dummy.tensor.set_(self.tensor.clone())
        return dummy

    def get_image_tensor(self):
        return self.tensor.squeeze(0)

    @th.no_grad()
    def set_image_tensor(self, tensor):
        self.tensor.set_(tensor.unsqueeze(0))

    @th.no_grad()
    def encode_image(self, pil_image, device='cuda', **kwargs):
        width, height = self.image_shape
        pil_image = pil_image.resize((width//self.scale, height//self.scale), Image.LANCZOS)
        self.tensor.set_(T.to_tensor(pil_image).unsqueeze(0).to(device, memory_format=th.channels_last))

    @th.no_grad()
    def encode_random(self):
        self.tensor.uniform_()
