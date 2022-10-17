
import numpy as np
import torch
import random


class RandomMaskingGenerator_base:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):


        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask # [196]



class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio, mask_prob):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.mask_ratio = mask_ratio
        self.mask_prob = mask_prob
        self.height, self.width = input_size
        self.num_patches = self.height * self.width

        

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):

    
        if self.mask_ratio[0] or self.mask_ratio[1]:
            if random.random() > self.mask_prob:
                self.num_mask = 0
            else:
                assert self.mask_ratio[1] >= self.mask_ratio[0]
                rand_ratio = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
                # rand_ratio = 0.5
                self.num_mask = int(rand_ratio * self.num_patches)

        else: # msak ratio = 0

            self.num_mask = 0

        
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)

        return mask # [196]

    