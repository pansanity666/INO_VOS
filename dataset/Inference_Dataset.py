"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import os
import torch
import cv2
import torch.utils.data as data
import numpy as np
import torchvision.transforms as tf

from tqdm import tqdm
from PIL import Image
from palette import custom_palette

class DLBase(data.Dataset):

    def __init__(self, *args, **kwargs):
        super(DLBase, self).__init__(*args, **kwargs)

        # RGB
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]

        self._init_means()

    def _init_means(self):
        self.MEAN255 = [255.*x for x in self.MEAN]
        self.STD255 = [255.*x for x in self.STD]

    def _init_palette(self, num_classes):
        self.palette = custom_palette(num_classes)

    def get_palette(self):
        return self.palette

    def remove_labels(self, mask):
        # Remove labels not in training
        for ignore_label in self.ignore_labels:
            mask[mask == ignore_label] = 255

        return mask.long()

def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

def read_frame(frame_dir, scale_size=[480], round_number=64):
    """
    read a single frame & preprocess
    """
    
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape

    if scale_size is not None:
        if len(scale_size) == 1:
            # set the shoter side as scale_size[0], keep ratio
            if(ori_h > ori_w):
                tw = scale_size[0]
                th = (tw * ori_h) / ori_w
                th = int((th // round_number) * round_number)
            else: 
                th = scale_size[0]
                tw = (th * ori_w) / ori_h # keep ratio scale
                tw = int((tw // round_number) * round_number) # make sure the image size can be divided by 64
        else:
            th, tw = scale_size
    else:
        # for extremely large frames in TYVOS, we mannuly resize to smaller size for preventing OOM. 
        if ori_h==1080 and ori_w==1920:
            th, tw = 720, 1280
        elif ori_h==816 and ori_w==1920:
            th, tw = 720, 1280
        else:
            th = int((ori_h // round_number) * round_number)
            tw = int((ori_w // round_number) * round_number ) 

    # original
    img_ori = img.astype(np.float32)
    img_ori = img_ori / 255.0
    img_ori = img_ori[:, :, ::-1]
    img_ori = np.transpose(img_ori.copy(), (2, 0, 1))
    img_ori = torch.from_numpy(img_ori).float()
    img_ori = color_normalize(img_ori)
    # print(img_ori.shape) 3, 480, 910

    # resize
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img)
    # print(img.shape) 3, 480, 896

    return img, img_ori

class DataSeg(DLBase):

    def __init__(self, args, split_list, ignore_labels=[], 
                 num_classes=7, root=os.path.expanduser('./data'), renorm=False):

        super(DataSeg, self).__init__()

        self.args = args
        self.num_classes = num_classes
        self.root = root
        self.ignore_labels = ignore_labels
        self._init_palette(self.num_classes)
    
        self.sequence_ids = [] # save accumulated frame number for each vid
        self.sequence_names = [] # save video names for each vid
        
        def add_sequence(name):
            vlen = len(self.images)
            self.sequence_ids.append(vlen)
            self.sequence_names.append(name)
            return vlen

        self.images = []
        self.masks = []
        self.flags = []

        token = None
        for line in tqdm(split_list):
            _flag, _image, _mask = line.strip("\n").split(' ')
            
            # save every frame
            #_flag = 1
            self.flags.append(int(_flag))

            assert os.path.isfile(_image), '%s not found' % _image

            _token = _image.split("/")[-2] # parent directory # token means video name token
            
            if token != _token:
                if not token is None:
                    add_sequence(token)
                token = _token
            self.images.append(_image)

            if _mask is None:
                self.masks.append(None)
            else:
                # _mask = os.path.join(cfg.DATASET.ROOT, _mask.lstrip('/'))
                # assert os.path.isfile(_mask), '%s not found' % _mask
                self.masks.append(_mask)

        # update the last sequence
        # returns the total amount of frames
        add_sequence(token)
        print("Loaded {} sequences".format(len(self.sequence_ids)))

        self.tf = tf.Compose([tf.ToTensor(), tf.Normalize(mean=self.MEAN, std=self.STD)])
        self._num_samples = len(self.images)

    def __len__(self):
        return len(self.sequence_ids)


    # def _mask2tensor(self, mask, num_classes=6):
    #     h,w = mask.shape
    #     ones = torch.ones(1,h,w)
    #     zeros = torch.zeros(num_classes,h,w)
        
    #     max_idx = mask.max()
    #     assert max_idx < num_classes, "{} >= {}".format(max_idx, num_classes)
    #     return zeros.scatter(0, mask[None, ...], ones)
    
    def denorm(self, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0,1,2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image

    def __getitem__(self, index):
        
        seq_to = self.sequence_ids[index] # end index
        seq_from = 0 if index == 0 else self.sequence_ids[index - 1]

        image0 = Image.open(self.images[seq_from])
        w,h = image0.size
        
        images_ori, images, masks, fns, flags = [], [], [], [], []
    
        tracks = torch.LongTensor(self.num_classes).fill_(-1) # torch.Size([7])
        masks = torch.LongTensor(self.num_classes, h, w).zero_() # torch.Size([7, 480, 910]) # mask keeps the same size as the original img, will be downscaled to the embed size later   
        known_ids = set()

        # loop of frames for current video
        for t in range(seq_from, seq_to):
            
            t0 = t - seq_from
            # image = Image.open(self.images[t]).convert('RGB')
            image, image_ori = read_frame(self.images[t], scale_size=self.args.scale_size, round_number=self.args.round_number)
            images.append(image)
            images_ori.append(image_ori)

            # pro
            # print(self.masks[t])
            if os.path.isfile(self.masks[t]):

                mask = Image.open(self.masks[t])
                mask = torch.from_numpy(np.array(mask, np.long, copy=False))
                unique_ids = np.unique(mask)
                
                # print(mask.shape) # 720, 1280
                # print(unique_ids) # 0, 1, 2, 3

                for oid in unique_ids: 
                    if not oid in known_ids: # if is the first time appearing of oid
                        tracks[oid] = t0 # track the first occurance of a segmentation category
                        known_ids.add(oid)
                        masks[oid] = (mask == oid).long() # category id to one-hot vector, first mask for each category 
            else:
                _image = Image.open(self.images[t]).convert('RGB')
                mask = Image.new('L', _image.size)

            fns.append(os.path.basename(self.images[t].replace(".jpg", "")))
            flags.append(self.flags[t])

        # 
        images = torch.stack(images, 0)
        images_ori = torch.stack(images_ori, 0)

        seq_name = self.sequence_names[index]
        flags = torch.LongTensor(flags)

        # print(images_ori.shape) # 69, 3, 480, 910
        # print(images.shape)  # 69, 3, 480, 896        
        # print(masks.shape) # NUM_CLASS, h, w; 7, 720, 1280 # first mask for each category 
        # print(tracks) # [ 0,  0,  0,  0, -1, -1, -1] # track the first occurance of a label
        # print(known_ids) # {0, 1, 2, 3}
        # print(flags) # [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, ... ]
      
        return images_ori, images, masks, tracks, len(known_ids), fns, flags, seq_name, 
