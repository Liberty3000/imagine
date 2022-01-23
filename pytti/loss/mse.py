import math, re
from PIL import Image
import torch as th
from torchvision.transforms import functional as T
from torch.nn import functional as F
from pytti.loss import Loss
from pytti.util import Rotoscoper, parse_prompt_string

class MSELoss(Loss):
    @th.no_grad()
    def __init__(self, comp, weight=0.5, stop=-math.inf, image_shape=None, name='direct target loss', device='cuda'):
        super().__init__(weight, stop, name)
        self.register_buffer('comp', comp)

        if image_shape is None:
            height, width = comp.shape[-2:]
            image_shape = (width, height)
        self.image_shape = image_shape

        self.register_buffer('mask', th.ones(1,1,1,1, device=device))
        self.use_mask = False

    @classmethod
    def convert_input(cls, input, img):
        return input

    @classmethod
    @th.no_grad()
    def TargetImage(cls, prompt_string, image_shape, pil_image=None, is_path=False, device='cuda'):
        text, weight, mask, stop = parse_prompt_string(prompt_string)

        if pil_image is None and text != '' and is_path:
            pil_image = Image.open(fetch(text)).convert('RGB')
            im = pil_image.resize(image_shape, Image.LANCZOS)
            comp = cls.make_comp(im)
        elif pil_image is None:
            comp = th.zeros(1,1,1,1, device=device)
        else:
            im = pil_image.resize(image_shape, Image.LANCZOS)
            comp = cls.make_comp(im)
        if image_shape is None:
            image_shape = pil_image.size
        output = cls(comp, weight, stop, '{} (direct)'.format(text), image_shape)
        output.set_mask(mask)
        return output

    @th.no_grad()
    def set_mask(self, mask, inverted = False, device='cuda'):
        if isinstance(mask, str) and mask != '':
            if mask[0] == '-':
                mask = mask[1:]
                inverted = True
            if mask.strip()[-4:] == '.mp4':
                r = Rotoscoper(mask, self)
                r.update(0)
                return
            mask = Image.open(fetch(mask)).convert('L')
        if isinstance(mask, Image.Image):
            mask = T.to_tensor(mask).unsqueeze(0).to(device, memory_format=th.channels_last)
        if mask not in ['',None]:
            self.mask.set_(mask if not inverted else (1-mask))
        self.use_mask = mask not in ['',None]

    @classmethod
    def make_comp(cls, pil_image, device='cuda'):
        out = T.to_tensor(pil_image).unsqueeze(0).to(device, memory_format=th.channels_last)
        return cls.convert_input(out, None)

    def set_comp(self, pil_image, device='cuda'):
        self.comp.set_(type(self).make_comp(pil_image))

    def get_loss(self, input, img):
        input = type(self).convert_input(input, img)
        if self.use_mask:
            if self.mask.shape[-2:] != input.shape[-2:]:
                with th.no_grad():
                    mask = T.resize(self.mask, input.shape[-2:])
                    self.set_mask(mask)
            return F.mse_loss(input*self.mask, self.comp * self.mask)
        else:
            return F.mse_loss(input, self.comp)
