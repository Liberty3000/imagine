import torch as th

class PaletteLoss(th.nn.Module):
    def __init__(self, n_palettes, weight=0.15, device='cuda'):
        super().__init__()
        self.n_palettes = n_palettes
        self.register_buffer('weight', th.as_tensor(weight).to(device))

    def forward(self, input):
        loss, loss_raw = 0,0
        if input.__class__.__name__ == 'PixelImage':
            tensor = input.tensor.movedim(0,-1).contiguous().view(-1, self.n_palettes).softmax(dim=-1)
            n,_ = tensor.shape
            mu = tensor.sub(tensor.mean(dim=0, keepdim = True))
            sigma = tensor.std(dim=0, keepdim = True)
            # SVD
            S = (tensor.transpose(0,1) @ tensor).div(sigma * sigma.transpose(0,1) * n)
            # minimize correlation (anti-correlate palettes)
            S.sub_(th.diag(S.diagonal()))
            loss_raw = S.mean()
            # maximze variance within each palette
            loss_raw.add_(sigma.mul(n).pow(-1).mean())
            loss = loss_raw * self.weight
        return loss, loss_raw

    @th.no_grad()
    def set_weight(self, weight, device = 'cuda'):
        self.weight.set_(th.as_tensor(weight, device=device))
