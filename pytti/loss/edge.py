import torch as th, torchvision as tv
from pytti.loss.mse import MSELoss

class EdgeLoss(MSELoss):

    @classmethod
    def convert_input(cls, input, img):
        return EdgeLoss.get_edges(input)

    @staticmethod
    def get_edges(tensor, device='cuda'):
        tensor = tv.transformers.functional.rgb_to_grayscale(tensor)
        kernel_1 = [[[[1,0,-1],[2,0,-2],[ 1, 0,-1]]]]
        kernel_2 = [[[[1,2, 1],[0,0, 0],[-1,-2,-1]]]]
        dx_ker = th.tensor(kernel_1).to(device=device, memory_format=th.channels_last).float().div(8)
        dy_ker = th.tensor(kernel_2).to(device=device, memory_format=th.channels_last).float().div(8)
        f_x = th.nn.functional.conv2d(tensor, dx_ker, padding='same')
        f_y = th.nn.functional.conv2d(tensor, dy_ker, padding='same')
        return th.cat([f_x,f_y])
