import torch as th
from neurpy.util import replace_grad
from pytti.util import parametric_eval

class Loss(th.nn.Module):
    def __init__(self, weight, stop, name):
        super().__init__()
        self.weight, self.stop, self.name = weight, stop, name
        self.input_axes = ('n','s','y','x')
        self.enabled = True

    def get_loss(self, input, img):
        raise NotImplementedError()

    def set_enabled(self, enabled):
        self.enabled = enabled

    def set_stop(stop):
        self.stop = stop

    def set_weight(weight):
        self.weight = weight

    def __str__(self):
        return self.name

    def forward(self, input, img, device='cuda'):
        if not self.enabled or self.weight in [0,'0']:
            return 0, 0

        weight = th.as_tensor(parametric_eval(self.weight), device=device)
        stop   = th.as_tensor(parametric_eval(self.stop),   device=device)
        loss_raw = self.get_loss(input, img)
        loss  =  loss_raw * weight.sign()
        return weight.abs() * replace_grad(loss, th.maximum(loss, stop)), loss_raw


from pytti.loss.depth import DepthLoss
from pytti.loss.edge import EdgeLoss
from pytti.loss.hsv import HSVLoss
from pytti.loss.latent import LatentLoss
from pytti.loss.mse import MSELoss
from pytti.loss.tv import TVLoss
losses = dict(depth=DepthLoss, edge=EdgeLoss, hsv=HSVLoss, latent=LatentLoss, mse=MSELoss, tv=TVLoss)

def build_loss(weight_name, weight, name, img, pil_target=None):
    prefix, suffix = weight_name.split('_', 1)
    if prefix in ['stabilize']: Loss = type(img).get_preferred_loss()
    else: Loss = losses[prefix]
    output = Loss.TargetImage(f"{prefix} {weight_name}:{weight}", img.image_shape, pil_target)
    output.set_enabled(pil_target is not None)
    return output
