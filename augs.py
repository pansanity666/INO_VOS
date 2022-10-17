import torchvision
import skimage
import utils
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
import math
import numbers

IMG_MEAN = (0.4914, 0.4822, 0.4465)
IMG_STD  = (0.2023, 0.1994, 0.2010)
NORM = [transforms.ToTensor(), 
        transforms.Normalize(IMG_MEAN, IMG_STD)]

# GRID_global_crops_size = 224   # 224, 256, 288    

class ParamsWrapper(object):
    # crop and return parameterss

    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, img):
        
        # return transformed img and params

        for t in self.transform_list:
            
            if isinstance(t, RandomResizedCropParams):
                img, params = t(img)
            else: 
                img = t(img)
        
        return img, params
        
def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))
        
def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class RandomResizedCropParams(torch.nn.Module):
    def __init__(self, size, scale, ratio=(3. / 4., 4. / 3.), interpolation=Image.BICUBIC):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.scale = scale 
        self.ratio = ratio
        self.interpolation = interpolation

    def get_params(self, img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped #NOTE: w/h!! 

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = _get_image_size(img)
        # print(img.size) # w, h
        # assert False
        # print(width,height)
        # assert False
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for kk in range(10):
            # print(kk)
            r = torch.empty(1).uniform_(scale[0], scale[1]).item()
            # print(scale[0], scale[1])
            # print(r)
            target_area = area * r
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            # print(area, aspect_ratio)
            # print('asd', w,h, width, height)
            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w
            # else:
                # print('fail!_{}'.format(kk))

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        # return crop and crop params
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        crop = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
    
        return crop, (i,j,h,w)




class DataAugmentationDINO(object):
    def __init__(self, args, global_crops_scale, local_crops_scale, local_crops_number):

        self.args = args
        self.flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # NOTE: ablation- using color aug or not
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        # ====== transformation for the local crops ====== 
        # local RRC + F&C + Normalize
        self.local_transfo = ParamsWrapper([
            RandomResizedCropParams(self.args.local_crops_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            self.normalize,
        ])

        # global RRC + Normalize (no color jitter is used)
        ratio = (3. / 4., 4. / 3.) 
        # self.random_resize_crop_params = RandomResizedCropParams(self.args.global_crops_size, scale=global_crops_scale, ratio=ratio, interpolation=Image.BICUBIC)
        self.global_transfo = ParamsWrapper([
            RandomResizedCropParams(self.args.global_crops_size, scale=global_crops_scale, ratio=ratio, interpolation=Image.BICUBIC),
            self.normalize
        ])

        # ======= 
        ori_size = self.args.ori_resize
        at_size = (int(ori_size[0]*self.args.at_downscale_factor), int(ori_size[1]*self.args.at_downscale_factor))

        # print(ori_size, at_size) # (720, 1280) (360, 640)
        self.resize_ori = transforms.Resize(ori_size)
        self.resize_at = transforms.Resize(at_size)

    def __call__(self, image):

        if isinstance(image, torch.Tensor):
            image = Image.fromarray(image.numpy())
        assert isinstance(image, Image.Image)

        # globals
        globs = []
        for _ in range(self.args.global_crops_number):
            globs.append(self.global_transfo(image)[0])
            
        # locals
        locals = []
        for _ in range(self.args.local_crops_number):
            locals.append(self.local_transfo(image)[0])

        crops = globs + locals
        
        return crops + [self.normalize(self.resize_at(image))]

        