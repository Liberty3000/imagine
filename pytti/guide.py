import math, tqdm
import pandas as pd, torch as th
from torch.nn import functional as F
from pytti.util import *


class ImageGuide():
    def __init__(self, image, embedder, lr=None, opt=None, **opt_args):
        self.image, self.embedder = image, embedder
        if lr is None: lr = image.lr
        opt_args['lr'] = lr

        self.opt_args = opt_args
        if opt is None:
            self.opt = th.optim.Adam(self.image.parameters(), **opt_args)
        else: self.opt = opt

        self.dataframe = []

    def update(self, i, stage_i):
        pass

    def set_optim(self, opt=None):
        if opt is not None: self.opt = opt
        else: self.opt = th.optim.Adam(self.image.parameters(), **self.opt_args)

    def clear_dataframe(self):
        self.dataframe = []

    def run_steps(self, n_steps, prompts, interp_prompts, loss_augs, stop=-math.inf,
                  interp_steps=0, i_offset=0, skipped_steps=0):
        for i in tqdm.tqdm(range(n_steps)):
            self.update(i + i_offset, i + skipped_steps)
            total_loss = self.train(i + skipped_steps, prompts=prompts, interp_prompts=interp_prompts,
                                    loss_augs=loss_augs, interp_steps=interp_steps)
            if total_loss <= stop: break
        return i+1

    def train(self, i, prompts, interp_prompts, loss_augs, interp_steps=0, save_loss=False):
        self.opt.zero_grad()
        z = self.image.decode_training_tensor()
        losses = []
        if self.embedder is not None:
            image_embeds, offsets, sizes = self.embedder(self.image, input = z)

        if i < interp_steps:
            t = i / interp_steps
            interp_losses = [prompt(format_input(image_embeds, self.embedder, prompt), \
                             format_input(offsets, self.embedder, prompt), \
                             format_input(sizes, self.embedder, prompt))[0]*(1-t) for prompt in interp_prompts]
        else:
            t = 1
            interp_losses = [0]

        prompt_losses = {prompt:prompt(format_input(image_embeds, self.embedder, prompt), \
                                       format_input(offsets, self.embedder, prompt), \
                                       format_input(sizes, self.embedder, prompt)) for prompt in prompts}

        aug_losses = {aug:aug(format_input(z, self.image, aug), self.image) for aug in loss_augs}

        image_losses = {aug:aug(self.image) for aug in self.image.image_loss()}

        losses, losses_raw = zip(*map(unpack_dict, [prompt_losses,aug_losses,image_losses]))
        losses = list(losses)
        total_loss = sum(map(lambda x:sum(x.values()),losses)) + sum(interp_losses)

        total_loss.backward()
        self.opt.step()
        self.image.update()

        return float(total_loss.item())
