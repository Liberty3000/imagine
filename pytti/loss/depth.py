import gc, math, os, sys
import torch as th
from torch.nn import functional as F
from torchvision.transforms import functional as T
from PIL import Image
from pytti.loss.mse import MSELoss


infer_helper = None
def AdaBins(path):
    global infer_helper
    cwd = os.getcwd()
    if infer_helper is None:
        try:
            sys.path.append(path)
            os.chdir(path)
            from infer import InferenceHelper
            infer_helper = InferenceHelper(dataset='nyu')
        except Exception as ex:
            print(ex)
            sys.exit(1)
        finally:
            os.chdir(cwd)
    return infer_helper


class DepthLoss(MSELoss):
    @classmethod
    def make_comp(cls, pil_image, device='cuda'):
        depth,_ = DepthLoss.get_depth(pil_image)
        return th.from_numpy(depth).to(device)

    @th.no_grad()
    def set_comp(self, pil_image):
        self.comp.set_(DepthLoss.make_comp(pil_image))
        if self.use_mask and self.mask.shape[-2:] != self.comp.shape[-2:]:
            self.mask.set_(T.resize(self.mask, self.comp.shape[-2:]))

    def get_loss(self, input, img, max_depth_area=500_000):
        height, width = input.shape[-2:]
        image_area = width * height
        if image_area > max_depth_area:
            depth_scale = math.sqrt(max_depth_area/image_area)
            height, width = int(height * depth_scale), int(width * depth_scale)
            depth_input = T.resize(input, (height, width), interpolation=T.InterpolationMode.BILINEAR)
            depth_resized = True
        else:
            depth_input = input
            depth_resized = False

        _, depth_map  = infer_helper.model(depth_input)
        depth_map = F.interpolate(depth_map, self.comp.shape[-2:], mode='bilinear', align_corners=True)
        return super().get_loss(depth_map, img)

    @staticmethod
    def get_depth(pil_image, adabins_dir=None, max_depth_area=500_000):

        infer_helper = AdaBins(adabins_dir if adabins_dir else os.environ['ADABINS_DIR'])

        # `max_depth_area` -> if the area of an image is above this, the depth model fails
        width, height = pil_image.size
        image_area = width * height
        if image_area > max_depth_area:
            depth_scale = math.sqrt(max_depth_area / image_area)
            height, width = int(height * depth_scale), int(width * depth_scale)
            depth_input = pil_image.resize((width, height), Image.LANCZOS)
            depth_resized = True
        else:
            depth_input = pil_image
            depth_resized = False

        gc.collect()
        th.cuda.empty_cache()

        _, depth_map = infer_helper.predict_pil(depth_input)

        gc.collect()
        th.cuda.empty_cache()

        return depth_map, depth_resized
