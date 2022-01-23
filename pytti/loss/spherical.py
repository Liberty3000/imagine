from torch.nn import functional as F

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return x.sub_(y).norm(dim=-1).div_(2).arcsin_().pow_(2).mul_(2)
