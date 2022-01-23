import torch as th
from torch.nn import functional as F

class HDRLoss(th.nn.Module):
    def __init__(self, palette_size, n_palettes, gamma=2.5, weight=0.15,
        magic_color=[[0.299,0.587,0.114]], device='cuda'):
        super().__init__()
        self.register_buffer('weight',th.as_tensor(weight).to(device))
        palette = th.linspace(0,1, palette_size).pow(gamma).view(palette_size, 1)
        self.register_buffer('comp', palette.repeat(1, n_palettes).to(device))
        self.magic_color = magic_color

    def forward(self, input):
        loss, loss_raw = 0,0
        if input.__class__.__name__ == 'PixelImage':
            palette = input.sort_palette()
            scale = palette.new_tensor([self.magic_color]).sqrt()
            color_norms = th.linalg.vector_norm(palette * scale, dim=-1)
            loss_raw = F.mse_loss(color_norms, self.comp)
            loss = loss_raw * self.weight
        return loss, loss_raw

    @th.no_grad()
    def set_weight(self, weight, device='cuda'):
        self.weight.set_(th.as_tensor(weight, device=device))

    def __str__(self):
        return 'HDR normalization'
