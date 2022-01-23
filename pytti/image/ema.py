import torch as th
from pytti.image import DifferentiableImage

class EMAImage(DifferentiableImage):
    def __init__(self, width, height, tensor ,decay):
        super().__init__(width, height)
        self.tensor, self.decay = th.nn.Parameter(tensor), decay
        self.register_buffer( 'biased', th.zeros_like(tensor))
        self.register_buffer('average', th.zeros_like(tensor))
        self.register_buffer(  'accum',  th.tensor(1.))
        self.update()

    @th.no_grad()
    def update(self):
        if not self.training: raise RuntimeError('update() should only be called during training.')
        self.accum.mul_(self.decay)
        self.biased.mul_(self.decay)
        self.biased.add_((1 - self.decay) * self.tensor)
        self.average.copy_(self.biased)
        self.average.div_(1 - self.accum)

    @th.no_grad()
    def reset(self):
        if not self.training: raise RuntimeError('reset() should only be called during training.')
        self.biased.set_(th.zeros_like(self.biased))
        self.average.set_(th.zeros_like(self.average))
        self.accum.set_(th.ones_like(self.accum))
        self.update()

    def decode_tensor(self):
        return self.decode(self.average)

    def decode_training_tensor(self):
        return self.decode(self.tensor)

    def decode(self, tensor):
        raise NotImplementedError
