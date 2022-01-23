import torch as th
from kornia.color import rgb_to_hsv
from pytti.loss.mse import MSELoss

class HSVLoss(MSELoss):
    @classmethod
    def convert_input(cls, input, img):
        return th.cat((input, rgb_to_hsv(input)[:,1:,...]), dim=1)
