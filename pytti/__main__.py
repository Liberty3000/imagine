import click, gc, glob, json, math, os, pprint, random, subprocess, re, types, warnings
from PIL import Image
import torch as th
from pytti.guide import ImageGuide
from pytti.image.pixel import PixelImage
from pytti.image.rgb import RGBImage
from pytti.image.vqgan import VQGANImage
from pytti.loss import *
from pytti.perceptor import MultiPerceptorCLIP
from pytti.prompt import parse_prompt
from pytti.transforms import *
from pytti.util import *
from neurpy.model.pretrained import load
from neurpy.util import enforce_reproducibility


@click.option(           '--experiment', default='pytti')
@click.option(               '--outdir', default='experiments/pytti')
@click.option(            '--namespace', default='default')
# input prompts
@click.option(                 '--text',                   type=str)
@click.option(               '--prefix', default='',       type=str)
@click.option(               '--suffix', default='',       type=str)
@click.option(        '--image_prompts', default='',       type=str)
# image initialization
@click.option(                    '--w', default=420,      type=int)
@click.option(                    '--h', default=380,      type=int)
@click.option(           '--init_image', default=None,     type=str)
@click.option(           '--init_video', default=None,     type=str)
# run duration
@click.option(                '--steps', default=2**13,    type=int, help='total steps per scene.')
@click.option(      '--steps_per_frame', default=50,       type=int, help='image model steps per frame.')
@click.option(         '--interp_steps', default=0,        type=int, help='interpolation steps per frame.')
@click.option(                  '--fps', default=12,       type=int, help='frames per second.')
# optimization parameters
@click.option(                   '--lr', default=0.1,      type=float)
@click.option(             '--reset_lr', default=True,     type=bool)
# image model
@click.option(            '--generator', default='vqgan_imagenet_f16_16384',
type=click.Choice(['vqgan_imagenet_f16_16384', 'vqgan_wikiart_f16_16384']))
@click.option(              '--palette', default='vqgan',  type=click.Choice(['vqgan','limited','unlimited']))
@click.option(           '--pixel_size', default=1,        type=int)
@click.option(            '--smoothing', default=0.1,      type=float)
@click.option(           '--n_palettes', default=8,        type=int)
@click.option(         '--palette_size', default=8,        type=int)
@click.option(         '--rand_palette', default=False,    is_flag=True)
@click.option(         '--show_palette', default=False,    is_flag=True)
@click.option(         '--lock_palette', default=False,    is_flag=True)
@click.option(       '--target_palette', default=None,     type=str)
@click.option('--palette_normalization', default=2e-1,     type=float)
@click.option(                '--gamma', default=1,        type=float)
@click.option(           '--hdr_weight', default=1e-2,     type=float)
# image subsampling
@click.option(           '--perceptors', default=['ViT-B/32','ViT-B/16','RN50x4'],
type=click.Choice(['ViT-B/32','ViT-B/16','RN50','RN50x4','RN50x16','RN101']), multiple=True)
@click.option(                 '--cuts', default=40,       type=int)
@click.option(              '--cut_pow', default=1,        type=float)
@click.option(           '--cut_border', default=0.25,     type=float)
@click.option(          '--border_mode', default='reflect',type=click.Choice(['circular','clamp','constant','reflect','replicate']))
# video animation
@click.option(              '--animate', default='3D',     type=click.Choice(['off','2D','3D','video']))
@click.option(             '--sampling', default='bicubic',type=click.Choice(['bicubic','bilinear','nearest']))
@click.option(               '--infill', default='smear',  type=click.Choice(['mirror','wrap','black','smear']))
@click.option(          '--lock_camera', default=True,     type=bool)
@click.option(        '--field_of_view', default=40,       type=int)
@click.option(           '--near_plane', default=1,        type=int)
@click.option(            '--far_plane', default=10_000,   type=int)
@click.option(         '--animate_init', default=150,      type=int)
@click.option(        '--animate_steps', default=50,       type=int)
@click.option(          '--translate_x', default='256 * sin(pi + t)')
@click.option(          '--translate_y', default=' 32 * cos(pi + t)')
@click.option(          '--translate_z', default='(50 + pi * t) * sin(t / 2 * pi)**2', help='only used if `animate` == `3D`.')
@click.option(            '--rotate_3d', default='[1, 0, 0, 7e-3]', help=\
'must be a [w,x,y,z] rotation (unit) quaternion. use `--rotate_3d=[1,0,0,0]` for no rotation.')
@click.option(            '--rotate_2d', default='0')
@click.option(            '--zoom_x_2d', default='0')
@click.option(            '--zoom_y_2d', default='0')
# motion tracking
@click.option(                '--video', default=None,  type=str)
@click.option(         '--frame_stride', default=1,     type=int)
@click.option(          '--encode_each', default=True,  type=bool)
# stabilization
@click.option(     '--stabilize_weight', default='1',   type=str)
@click.option(      '--stabilize_depth', default='1',   type=str)
@click.option(       '--stabilize_edge', default='1',   type=str)
@click.option(   '--stabilize_semantic', default='0',   type=str)
# output specification
@click.option(               '--outdir', default=os.getcwd())
@click.option(           '--save_every', default=10)
@click.option(       '--save_embedding', default=False, is_flag=True)
# video compilation
@click.option(                '--video', default=False, is_flag=True)
@click.option(          '--keep_frames', default=False, is_flag=True)
# run reproducibility
@click.option(                 '--seed', default=random.randint(0,1e6))
@click.option(    '--non_deterministic', default=False, is_flag=True)
# device strategy
@click.option(               '--device', default='cuda')
@click.command()
@click.pass_context
def run(ctx, **args):
    assert th.cuda.is_available(), 'ERROR <!> :: CUDA not available.'
    assert args['text'] not in ['', None], 'ERROR <!> :: no input.'
    print('-' * 80)
    pprint.pprint(args, indent=2)
    print('-' * 80)
    conf = types.SimpleNamespace(**args)
    ############################################################################
    enforce_reproducibility(conf.seed)
    clear_rotoscopers()
    ############################################################################
    # load perceptors
    perceptors = [load(p)[0].to(conf.device) for p in conf.perceptors]
    embedder = MultiPerceptorCLIP(perceptors=perceptors, cutn=conf.cuts,
    cut_pow=conf.cut_pow, padding=conf.cut_border, border_mode=conf.border_mode)
    ############################################################################
    # load generator
    if conf.palette == 'vqgan':
        vqgan = load(conf.generator, image_size=(conf.h, conf.w))[0]
        img = VQGANImage(conf.w, conf.h, conf.pixel_size, model=vqgan.model)
        img.encode_random()
        conf.pixel_size = 1

    elif conf.palette == 'unlimited':
        img = RGBImage(conf.w, conf.h, conf.pixel_size)
        img.encode_random()

    elif conf.palette == 'limited':
        img = PixelImage(w=conf.w, h=conf.h, pixel_size=conf.pixel_size, n_palettes=conf.n_palettes,
        palette_size=conf.palette_size, gamma=conf.gamma, hdr_weight=conf.hdr_weight,
        palette_normalization=conf.palette_normalization)

        img.encode_random(random_palette=conf.rand_palette)

        if conf.target_palette is not None:
            target_palette = Image.open(conf.target_palette).convert('RGB')
            img.set_palette_target(target_palette)
        else:
            img.lock_palette(conf.lock_palette)

    else: raise NotImplementedError()
    ############################################################################
    # encode text
    scene_delimiter, prompt_delimiter = '|','||'
    prompts = [[parse_prompt(embedder, perceptors, prompt.strip()) for
    prompt in (conf.prefix + stage + conf.suffix).strip().split(scene_delimiter)
    if prompt.strip()] for stage in conf.text.split(prompt_delimiter) if stage]
    ############################################################################
    # set up file namespace
    os.makedirs(os.path.join(conf.outdir, f'{conf.namespace}'), exist_ok=True)
    expr = f'^(?P<pre>{re.escape(conf.namespace)}\\(?)(?P<index>\\d*)(?P<post>\\)?_1\\.png)$'
    comp = [f'{conf.namespace}_1.png', f'{conf.namespace}(1)_1.png']
    _, i = get_next_file(os.path.join(conf.outdir, f'{conf.namespace}'), expr, comp)
    output = conf.namespace if i == 0 else f'{conf.namespace}({i})'
    ############################################################################
    # load init image
    if conf.init_image is not None:
        print(f'loading {conf.init_image}...')
        init_image = Image.open(conf.init_image).convert('RGB')
        init_size = init_image.size
        # automatic aspect ratio matching
        if conf.w == -1:
            conf.w = int(conf.h * init_size[0] / init_size[1])
        if conf.h == -1:
            conf.h = int(conf.w * init_size[1] / init_size[0])
    else:
        init_image = None

    # load from video
    if conf.animate == 'video':
        print(f'loading {conf.init_video}...')
        video_frames = get_frames(conf.init_video)
        conf.animate_init = max(conf.steps, conf.animate_init)
        if init_image is None:
            init_frame = video_frames.get_data(0)
            init_image = Image.fromarray(init_frame).convert('RGB')
            init_size = init_image.size
            if conf.w == -1: conf.w = int(conf.h * init_size[0] / init_size[1])
            if conf.h == -1: conf.h = int(conf.w * init_size[1] / init_size[0])
    ############################################################################
    # configure loss
    loss_augs = []

    # image init
    if init_image is not None:
        img.encode_image(init_image)

        init_augs, name = ['direct_init_weight'], f'init image ({conf.init_image})'
        init_augs = [build_loss(aug, vars(conf)[aug], name, img, init_image) \
                     for aug in init_augs if vars(conf)[x] not in ['','0',0]]
        loss_augs.extend(init_augs)

        if conf.semantic_init_weight not in ['','0',0]:
            name = f'init image [{conf.init_image}]:{conf.semantic_init_weight}'
            semantic_init_prompt = parse_prompt(embedder, name, init_image)
            prompts[0].append(semantic_init_prompt)
        else:
            semantic_init_prompt = None
    else:
        init_augs, semantic_init_prompt = [], None
    ############################################################################
    # prompt losses
    loss_augs.extend(type(img).get_preferred_loss().TargetImage(p.strip(), img.image_shape, is_path=True)
    for p in conf.image_prompts.split(scene_delimiter) if p.strip())

    # stabilization losses
    stabilization_augs = ['stabilize_weight', 'stabilize_depth', 'stabilize_edge']
    stabilization_augs = [build_loss(aug, vars(conf)[aug], 'stabilize', img, init_image) \
    for aug in stabilization_augs if vars(conf)[aug] not in ['','0',0]]
    loss_augs.extend(stabilization_augs)

    if conf.stabilize_semantic not in ['','0',0]:
        arg = f'stabilize:{conf.stabilize_semantic}'
        last_frame_semantic = parse_prompt(embedder, arg, init_image if init_image else img.decode_image())
        last_frame_semantic.set_enabled(init_image is not None)
        for scene in prompts: scene.append(last_frame_semantic)
    else:
        last_frame_semantic = None

    # total variation loss
    if conf.smoothing != 0: loss_augs.append(TVLoss(weight=conf.smoothing))
    ############################################################################
    i = 0
    model = ImageGuide(img, embedder, lr=conf.lr)

    def update(i, stage_i):
        # save
        if i > 0 and conf.save_every > 0 and i % conf.save_every == 0:
            try: im
            except NameError: im = img.decode_image()

            n = i // conf.save_every
            d = str(i).zfill(6)
            fname = f'{conf.namespace}/{output}_{d}.png'
            im.save(os.path.join(conf.outdir, fname))

        t = (i - conf.animate_init) / (conf.steps_per_frame * conf.fps)
        set_t(t)
        if i >= conf.animate_init:
            if (i - conf.animate_init) % conf.steps_per_frame == 0:
                update_rotoscopers(((i - conf.animate_init) // conf.steps_per_frame + 1) * conf.frame_stride)

                if conf.reset_lr: model.set_optim(None)

                if conf.animate == '2D':
                    next_step_pil = zoom_2d(img, (conf.translate_x, conf.translate_y),
                    (conf.zoom_x_2d, conf.zoom_y_2d), conf.rotate_2d,
                    border_mode=conf.infill, sampling=conf.sampling)

                elif conf.animate == '3D':
                    try: im
                    except NameError: im = img.decode_image()

                    flow, next_step_pil = zoom_3d(img,
                    (conf.translate_x, conf.translate_y, conf.translate_z),
                    conf.rotate_3d, conf.field_of_view, conf.near_plane, conf.far_plane,
                    border_mode=conf.infill, sampling_mode=conf.sampling,
                    stabilize=conf.lock_camera)

                elif conf.animate == 'video': raise NotImplementedError()

                if conf.animate != 'off':
                    for aug in init_augs: aug.set_enabled(False)
                    for aug in stabilization_augs:
                        aug.set_comp(next_step_pil)
                        aug.set_enabled(True)
                    if semantic_init_prompt is not None:
                        semantic_init_prompt.set_enabled(False)
                    if last_frame_semantic is not None:
                        last_frame_semantic.set_image(embedder, next_step_pil)
                        last_frame_semantic.set_enabled(True)

    ############################################################################
    model.update = update
    export_args(vars(conf), f'{conf.outdir}/{conf.namespace}/{output}-args.txt')

    skip_steps   = i %  conf.steps
    skip_prompts = i // conf.steps
    last_scene = prompts[0] if skip_prompts == 0 else prompts[skip_prompts - 1]
    for scene in prompts[skip_prompts:]:
        steps = conf.steps - skip_steps
        i += model.run_steps(steps, scene, last_scene, loss_augs,
        interp_steps=conf.interp_steps, i_offset=i, skipped_steps=skip_steps)
        last_scene, skip_steps = scene, 0


if __name__ == '__main__':
    run()
