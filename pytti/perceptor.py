import kornia.augmentation as K
import torch as th, torchvision as tv
from torch.nn import functional as F
from pytti.util import *

class MultiPerceptorCLIP(th.nn.Module):
    def __init__(self, perceptors, cutn=32, cut_pow=1.5, padding=0.25, border_mode='clamp', noise=0.1):
        super().__init__()

        self.perceptors = perceptors
        self.normalize = tv.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954,0.26130258, 0.27577711])
        self.cut_sizes = [p.visual.input_resolution for p in self.perceptors]
        self.cutn, self.cut_pow, self.noise = cutn, cut_pow, noise
        self.padding, self.border_mode= padding, border_mode
        self.input_axes, self.output_axes = ('n', 's', 'y', 'x'), ('c', 'n', 'i')

        self.augs = th.nn.Sequential(
        K.RandomHorizontalFlip(p=0.3),
        K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
        K.RandomPerspective(0.2, p=0.4,),
        K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        K.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=False, p=0.7),
        th.nn.Identity())

    def make_cutouts(self, input, side_x, side_y, cut_size, device='cuda'):
        cutouts, offsets, sizes = [],[],[]
        min_size, max_size  = min(side_x, side_y, cut_size), min(side_x, side_y)
        paddingx = min(round(side_x * self.padding), side_x)
        paddingy = min(round(side_y * self.padding), side_y)

        for _ in range(self.cutn):
            size = int(max_size * (th.zeros(1,).normal_(mean=.8, std=.3).clip(cut_size/max_size, 1.) ** self.cut_pow))
            offsetx_max = side_x - size + 1
            offsety_max = side_y - size + 1


            if self.border_mode == 'clamp':
                offsetx = th.clamp((th.rand([])*(offsetx_max+2*paddingx) - paddingx).floor().int(), 0, offsetx_max)
                offsety = th.clamp((th.rand([])*(offsety_max+2*paddingy) - paddingy).floor().int(), 0, offsety_max)
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            else:
                px = min(size, paddingx)
                py = min(size, paddingy)
                offsetx = (th.rand([])*(offsetx_max+2*px) - px).floor().int()
                offsety = (th.rand([])*(offsety_max+2*py) - py).floor().int()
                cutout = input[:, :, paddingy + offsety:paddingy + offsety + size, paddingx + offsetx:paddingx + offsetx + size]

            cutouts.append(F.adaptive_avg_pool2d(cutout, cut_size))
            offsets.append(th.as_tensor([[offsetx/side_x, offsety/side_y]]).to(device))
            sizes.append(th.as_tensor([[size/side_x, size/side_y]]).to(device))

        cutouts = self.augs(th.cat(cutouts))
        offsets = th.cat(offsets)
        sizes   = th.cat(sizes)

        if self.noise:
            coef = cutouts.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise)
            cutouts.add_(coef * th.randn_like(cutouts))

        return cutouts, offsets, sizes

    def forward(self, diff_image, input = None, device = 'cuda'):
        side_x, side_y = diff_image.image_shape

        if input is None:
            input = format_module(diff_image, self).to(device=device, memory_format=th.channels_last)
        else:
            input = format_input(input, diff_image, self).to(device=device, memory_format=th.channels_last)

        max_size = min(side_x, side_y)
        image_embeds, all_offsets, all_sizes = [],[],[]

        paddingx = min(round(side_x * self.padding), side_x)
        paddingy = min(round(side_y * self.padding), side_y)

        if self.border_mode != 'clamp':
            padding = (paddingx, paddingx, paddingy, paddingy)
            input = F.pad(input, padding, mode=self.border_mode)

        for cut_size, perceptor in zip(self.cut_sizes, self.perceptors):
            cutouts, offsets, sizes = self.make_cutouts(input, side_x, side_y, cut_size)
            embs = perceptor.encode_image(self.normalize(cutouts)).float().unsqueeze(0)
            image_embeds.append(embs)
            all_offsets.append(offsets)
            all_sizes.append(sizes)

        return cat_with_pad(image_embeds), th.stack(all_offsets), th.stack(all_sizes)
