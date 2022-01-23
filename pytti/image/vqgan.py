from PIL import Image
import torch as th
from torch.nn import functional as F
from torchvision.transforms import functional as T
from pytti.image.ema import EMAImage
from pytti.util import format_module
from neurpy.noise.perlin import random_perlin
from neurpy.noise.pyramid import random_pyramid
from neurpy.util import replace_grad, clamp_with_grad


def vector_quantize(x, codebook, fake_grad = True):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class VQGANImage(EMAImage):
    def __init__(self, width, height, model, lr=0.1, ema_decay=0.99, latent_strength=1.):
        if 'Gumbel' in model.__class__.__name__:
            e_dim = 256
            n_toks = model.quantize.n_embed
            vqgan_quantize_embedding = model.quantize.embed.weight
        else:
            e_dim = model.quantize.e_dim
            n_toks = model.quantize.n_e
            vqgan_quantize_embedding = model.quantize.embedding.weight

        self.f = 2**(model.decoder.num_resolutions - 1)
        self.e_dim = e_dim
        self.n_toks = n_toks
        toksX, toksY = width // self.f, height // self.f
        sideX, sideY = toksX * self.f, toksY * self.f
        self.toksX, self.toksY = toksX, toksY

        z = self.random_latent(vqgan_quantize_embedding=vqgan_quantize_embedding)
        super().__init__(sideX, sideY, z, ema_decay)

        self.output_axes = ('n', 's', 'y', 'x')
        self.lr, self.latent_strength = lr, latent_strength
        self.register_buffer('vqgan_quantize_embedding', vqgan_quantize_embedding, persistent=False)
        self.vqgan_decode = model.decode
        self.vqgan_encode = model.encode

    def clone(self):
        dummy = VQGANImage(*self.image_shape)
        with torch.no_grad():
            dummy.tensor.set_(self.tensor.clone())
            dummy.accum.set_(self.accum.clone())
            dummy.biased.set_(self.biased.clone())
            dummy.average.set_(self.average.clone())
        dummy.decay = self.decay
        return dummy

    def get_latent_tensor(self, detach=False, device='cuda'):
        z = self.tensor
        if detach: z = z.detach()
        z_q = vector_quantize(z, self.vqgan_quantize_embedding).movedim(3, 1).to(device)
        return z_q

    @classmethod
    def get_preferred_loss(cls):
        from pytti.loss.latent import LatentLoss
        return LatentLoss

    def decode(self, z, device='cuda'):
        z_q = vector_quantize(z, self.vqgan_quantize_embedding).movedim(3, 1).to(device)
        output = self.vqgan_decode(z_q).add(1).div(2)
        width, height = self.image_shape
        return clamp_with_grad(output, 0, 1)

    @th.no_grad()
    def encode_image(self, pil_image, device='cuda', **kwargs):
        pil_image = pil_image.resize(self.image_shape, Image.LANCZOS)
        pil_image = T.to_tensor(pil_image)
        z, *_ = self.vqgan_encode(pil_image.unsqueeze(0).to(device) * 2 - 1)

        self.tensor.set_(z.movedim(1,3))
        self.reset()

    @th.no_grad()
    def make_latent(self, pil_image, device='cuda'):
        pil_image = pil_image.resize(self.image_shape, Image.LANCZOS)
        pil_image = T.to_tensor(pil_image)
        z, *_ = self.vqgan_encode(pil_image.unsqueeze(0).to(device) * 2 - 1)

        z_q = vector_quantize(z.movedim(1,3), self.vqgan_quantize_embedding).movedim(3, 1).to(device)
        return z_q

    @th.no_grad()
    def encode_random(self, init='random', *args, **kwargs):
        if init == 'random':
            self.tensor.set_(self.random_latent())
        elif init == 'perlin':
            self.tensor.set_(self.perlin_latent(kwargs['perlin_weight'], kwargs['perlin_octaves']))
        elif init == 'pyramid':
            self.tensor.set_(self.pyramid_latent(kwargs['pyramid_octaves'], kwargs['pyramid_decay']))
        else: raise NotImplementedError()
        self.reset()

    def random_latent(self, device='cuda', vqgan_quantize_embedding=None):
        if vqgan_quantize_embedding is None: vqgan_quantize_embedding = self.vqgan_quantize_embedding
        n_toks = self.n_toks
        toksX, toksY = self.toksX, self.toksY
        one_hot = F.one_hot(th.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        z = one_hot @ vqgan_quantize_embedding
        z = z.view([-1, toksY, toksX, self.e_dim])
        return z

    def perlin_latent(self, perlin_weight, perlin_octaves, device='cuda'):
        rand_init = random_perlin((self.toksY * self.f, self.toksX * self.f), perlin_weight, perlin_octaves)
        z, *_ = self.vqgan_encode(rand_init.to(device) * 2 - 1)
        z = z.permute(0, 2, 3, 1)
        return z

    def pyramid_latent(self, pyramid_octaves, pyramid_decay, device='cuda'):
        rand_init = random_pyramid((1, 3, self.toksY * self.f, self.toksX * self.f), pyramid_octaves, pyramid_decay)
        rand_init = (rand_init * 0.5 + 0.5).clip(0, 1)
        z, *_ = self.vqgan_encode(rand_init.to(device) * 2 - 1)
        z = z.permute(0, 2, 3, 1)
        return z
