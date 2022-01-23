import copy, gc, math, os, re
import torch as th
from torchvision.transforms import functional as T
from torch.nn import functional as F
from PIL import Image
from pytti.loss.mse import MSELoss
from pytti.util import parse_prompt_string

class LatentLoss(MSELoss):
    @th.no_grad()
    def __init__(self, comp, weight=0.5, stop=-math.inf, name='direct target loss', image_shape=None):
        super().__init__(comp, weight, stop, name, image_shape)
        w, h = image_shape
        self.image_shape = image_shape
        self.pil_image, self.has_latent = None, False
        self.loss_fn = MSELoss(T.resize(comp.clone(), (h,w)), weight, stop, name, image_shape)

    @classmethod
    @th.no_grad()
    def TargetImage(cls, prompt_string, image_shape, pil_image = None, is_path = False, device = 'cuda'):
        text, weight, mask, stop = parse_prompt_string(prompt_string)

        if pil_image is None and text != '' and is_path:
            pil_image = Image.open(fetch(text)).convert('RGB')
        comp = MSELoss.make_comp(pil_image) if pil_image is not None else th.zeros(1,1,1,1, device=device)
        output = cls(comp, weight, stop, '{} (latent)'.format(text), image_shape)
        if pil_image is not None:
            output.set_comp(pil_image)
        output.set_mask(mask)
        return output

    @th.no_grad()
    def set_comp(self, pil_image, device='cuda'):
        self.pil_image, self.has_latent = pil_image, False
        self.loss_fn.set_comp(self.pil_image.resize(self.image_shape, Image.LANCZOS))

    def set_mask(self, mask, inverted = False):
        self.loss_fn.set_mask(mask, inverted)
        super().set_mask(mask, inverted)

    def get_loss(self, input, img):
        if not self.has_latent:
            latent = img.make_latent(self.pil_image)
            with th.no_grad():
                self.comp.set_(latent.clone())
            self.has_latent = True
        l1 = super().get_loss(img.get_latent_tensor() ,img)/2
        l2 = self.loss_fn.get_loss(input, img)/10
        return l1 + l2
