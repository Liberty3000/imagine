import math, torch
from torch.nn import functional as F
from pytti.loss import Loss

def total_variation_loss(img, weight=1.):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = th.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = th.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

# l2 total variation loss, as in Mahendran et al.
def tv_loss(input):
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])

class TVLoss(Loss):
    def __init__(self, weight=0.15, stop=-math.inf, name='total variation loss'):
        super().__init__(weight, stop, name)

    def get_loss(self, input, img):
        return tv_loss(input)
