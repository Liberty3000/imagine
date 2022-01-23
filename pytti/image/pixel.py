import math, numpy as np, torch as th
from torch.nn import functional as F
from torchvision.transforms import functional as T
from PIL import Image
from pytti.guide import ImageGuide
from pytti.image import DifferentiableImage
from pytti.loss.hdr import HDRLoss
from pytti.loss.hsv import HSVLoss
from pytti.loss.palette import PaletteLoss
from pytti.util import break_tensor, closest_color
from neurpy.util import replace_grad


class PixelImage(DifferentiableImage):
    def __init__(self, w, h, pixel_size, palette_size, n_palettes, palette_normalization=1e-1, palette_inertia=2, gamma=1., hdr_weight=0.5, latent_strength=1e-1, magic_color=[[[0.299]],[[0.587]],[[0.114]]], device='cuda'):
        super().__init__(w * pixel_size, h * pixel_size)

        self.values = th.nn.Parameter(th.zeros(h, w).to(device))
        self.tensor = th.nn.Parameter(th.zeros(n_palettes, h, w).to(device))

        init = th.linspace(0,palette_inertia,palette_size).pow(gamma)
        self.palette = th.nn.Parameter(init.view(palette_size,1,1).repeat(1,n_palettes,3).to(device))

        self.output_axes = ('n', 's', 'y', 'x')
        self.pixel_size, self.palette_inertia = pixel_size, palette_inertia
        self.n_palettes, self.palette_size = palette_size, n_palettes
        self.latent_strength, self.magic_color = latent_strength, magic_color

        self.hdr_loss = HDRLoss(palette_size, n_palettes, gamma, hdr_weight) if hdr_weight != 0 else None
        self.palette_loss = PaletteLoss(n_palettes, palette_normalization)

        self.register_buffer('palette_target', th.empty_like(self.palette))
        self.use_palette_target = False

    def clone(self):
        w, h = self.image_shape
        dummy = PixelImage(w // self.pixel_size, h // self.pixel_size,
                           self.pixel_size, self.palette_size, self.n_palettes,
                           palette_normalization=float(self.palette_loss.weight),
                           hdr_weight=0 if self.hdr_loss is None else float(self.hdr_loss.weight))
        with th.no_grad():
            dummy.value.set_(self.values.clone())
            dummy.tensor.set_(self.tensor.clone())
            dummy.palette.set_(self.palette.clone())
            dummy.palette_target.set_(self.palette_target.clone())
            dummy.use_palette_target = self.use_palette_target
        return dummy

    def get_image_tensor(self):
        return th.cat([self.values.unsqueeze(0), self.tensor])

    @th.no_grad()
    def set_image_tensor(self, tensor):
        self.values.set_(tensor[0])
        self.tensor.set_(tensor[1:])

    def image_loss(self):
        return [_ for _ in [self.hdr_loss, self.palette_loss] if _ is not None]

    @th.no_grad()
    def lock_palette(self, lock = True):
        if lock: self.palette_target.set_(self.sort_palette().clone())
        self.use_palette_target = lock


    def set_palette_target(self, pil_image):
        if pil_image is None:
            self.use_palette_target = False
        else:
            dummy = self.clone()
            dummy.use_palette_target = False
            dummy.encode_image(pil_image)
            with th.no_grad():
                self.palette_target.set_(dummy.sort_palette())
                self.palette.set_(self.palette_target.clone())
                self.use_palette_target = True

    def sort_palette(self):
        if self.use_palette_target: return self.palette_target
        palette = (self.palette/self.palette_inertia).clamp_(0,1)
        magic_color = palette.new_tensor([[[0.299,0.587,0.114]]])
        color_norms = palette.square().mul_(magic_color).sum(dim = -1)
        palette_indices = color_norms.argsort(dim = 0).T
        palette = th.stack([palette[i][:,j] for j,i in enumerate(palette_indices)],dim=1)
        return palette

    @th.no_grad()
    def render_palette(self):
        palette = self.sort_palette()
        w, h = self.n_palettes*16, self.palette_size*32
        array = np.array(palette.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8))[:,:,:]
        return Image.fromarray(array).resize((w,h), Image.NEAREST)

    def decode_tensor(self):
        w, h = self.image_shape
        palette = self.sort_palette()

        values = self.values.clamp(0,1) * (self.palette_size-1)
        value_floors, value_ceils, value_rounds, value_fracs = break_tensor(values)
        value_fracs = value_fracs.unsqueeze(-1).unsqueeze(-1)

        palette_weights = self.tensor.movedim(0,2)
        palettes = F.one_hot(palette_weights.argmax(dim = 2), num_classes=self.n_palettes)
        palette_weights = palette_weights.softmax(dim = 2).unsqueeze(-1)
        palettes = palettes.unsqueeze(-1)

        colors_disc = palette[value_rounds]
        colors_disc = (colors_disc * palettes).sum(dim = 2)
        colors_disc = colors_disc.movedim(2,0).unsqueeze(0).to('cuda', memory_format=th.channels_last)
        colors_disc = F.interpolate(colors_disc, (h, w) , mode='nearest')

        colors_cont = (palette[value_floors]*(1-value_fracs) + palette[value_ceils]*value_fracs)
        colors_cont = (colors_cont * palette_weights).sum(dim = 2)
        colors_cont = colors_cont.movedim(2,0).unsqueeze(0).to('cuda', memory_format=th.channels_last)
        colors_cont = F.interpolate(colors_cont, (h, w) , mode='nearest')
        return replace_grad(colors_disc, colors_cont*0.5+colors_disc*0.5)


    @th.no_grad()
    def render_value_image(self):
        w, h = self.image_shape
        values = self.values.clamp(0,1).unsqueeze(-1).repeat(1,1,3)
        array = np.array(values.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8))[:,:,:]
        return Image.fromarray(array).resize((w,h), Image.NEAREST)


    @th.no_grad()
    def render_channel(self, palette_i):
        w, h = self.image_shape
        palette = self.sort_palette()
        palette[:,:palette_i   ,:] = 0.5
        palette[:, palette_i+1:,:] = 0.5

        values = self.values.clamp(0,1)*(self.palette_size-1)
        value_floors, value_ceils, value_rounds, value_fracs = break_tensor(values)
        value_fracs = value_fracs.unsqueeze(-1).unsqueeze(-1)

        palette_weights = self.tensor.movedim(0,2)
        palettes = F.one_hot(palette_weights.argmax(dim = 2), num_classes=self.n_palettes)
        palette_weights = palette_weights.softmax(dim = 2).unsqueeze(-1)

        colors_cont = palette[value_floors]*(1-value_fracs) + palette[value_ceils]*value_fracs
        colors_cont = (colors_cont * palette_weights).sum(dim = 2)
        colors_cont = F.interpolate(colors_cont.movedim(2,0).unsqueeze(0), (h, w) , mode='nearest')

        tensor = named_rearrange(colors_cont, self.output_axes, ('y', 'x', 's'))
        array = tensor.mul(255).clamp(0, 255).cpu().detach().numpy()
        return Image.fromarray(np.array(array.astype(np.uint8)[:,:,:]))


    @th.no_grad()
    def update(self):
        self.palette.clamp_(0,self.palette_inertia)
        self.values.clamp_(0,1)
        self.tensor.clamp_(0,float('inf'))


    def encode_image(self, pil_image, steps=200, lr=1e-1, device='cuda'):
        w, h = self.image_shape
        pixel_size = self.pixel_size
        color_ref = pil_image.resize((w // pixel_size, h // pixel_size), Image.LANCZOS)
        color_ref = TF.to_tensor(color_ref).to(device)
        with th.no_grad():
            magic_color = self.palette.new_tensor(self.magic_color)
            value_ref = th.linalg.vector_norm(color_ref * (magic_color.sqrt()), dim=0)
            self.values.set_(value_ref)

        mse = HSVLoss.TargetImage('HSV loss', self.image_shape, pil_image)

        if self.hdr_loss is not None:
            before_weight = self.hdr_loss.weight.detach()
            self.hdr_loss.set_weight(0.01)

        opt = th.optim.Adam([self.palette, self.tensor], lr=lr)
        guide = DirectImageGuide(self, None, opt=opt)
        guide.run_steps(steps,[],[],[mse])

        if self.hdr_loss is not None: self.hdr_loss.set_weight(before_weight)


    @th.no_grad()
    def encode_random(self, random_palette = False):
        self.values.uniform_()
        self.tensor.uniform_()
        if random_palette: self.palette.uniform_(to=self.palette_inertia)
