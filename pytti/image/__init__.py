import copy
import torch as th, numpy as np
from PIL import Image
from pytti.util import named_rearrange

class DifferentiableImage(th.nn.Module):
    def __init__(self, width, height, pixel_format='RGB', lr=2e-2):
        super().__init__()
        error = f'PIL image mode {pixel_format} is not supported.'
        assert pixel_format in ['L','RGB','I','F'], error
        self.image_shape = (width, height)
        self.lr = lr
        self.pixel_format = format
        self.output_axes = ('x', 'y', 's')
        self.latent_strength = 0

    def get_image_tensor(self):
        raise NotImplementedError

    def set_image_tensor(self, tensor):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

    def encode_image(self, pil_image):
        raise NotImplementedError

    def encode_random(self):
        raise NotImplementedError

    def decode_tensor(self):
        raise NotImplementedError

    def decode_training_tensor(self):
        return self.decode_tensor()

    def decode_image(self):
        tensor = self.decode_tensor()
        tensor = named_rearrange(tensor, self.output_axes, ('y', 'x', 's'))
        array = tensor.mul(255).clamp(0, 255).cpu().detach().numpy()
        pil_image = Image.fromarray(array.astype(np.uint8)[:,:,:])
        return pil_image

    def get_latent_tensor(self, detach=False):
        if detach:
            return self.get_image_tensor().detach()
        else:
            return self.get_image_tensor()

    def make_latent(self, pil_image):
        try: dummy = self.clone()
        except NotImplementedError: dummy = copy.deepcopy(self)
        dummy.encode_image(pil_image)
        return dummy.get_latent_tensor(detach=True)

    def image_loss(self):
        return []

    @classmethod
    def get_preferred_loss(cls):
        from pytti.loss.hsv import HSVLoss
        return HSVLoss

    def forward(self):
        if not self.training: return self.decode_tensor()
        else: return self.decode_training_tensor()

    def update(self):
        pass

from pytti.image.rgb import RGBImage
from pytti.image.pixel import PixelImage
from pytti.image.vqgan import VQGANImage
