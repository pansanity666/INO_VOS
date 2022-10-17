import torch
import torchvision.io as tio
import os 
import torch.utils.data 

from tqdm import tqdm
from PIL import Image
import sys

# Multi-thread script for preparing the neighbour mask offline. (Used for fast label propagation)

DATASET = sys.argv[1]

if DATASET == 'davis':
    VID_PATH = './data/DAVIS/JPEGImages/480p'
    PATCH_SIZE = 8
    ROUND_NUMBER = [8]
    SIZE_MASK_NEIGHBORHOOD = 40 # size mask neighborhood
    scale_size = [480] # 
    
elif DataSeg == 'ytvos':
    VID_PATH = './data/YouTube_VOS/valid_all_frames/JPEGImages'
    PATCH_SIZE = 8
    ROUND_NUMBER = [8]
    SIZE_MASK_NEIGHBORHOOD = 50 # size mask neighborhood
    scale_size = None
else:
    assert False, 'Only support davis and ytvos A.T.M.'
    
MSK_SAVE_PATH = './cached/masks'
os.makedirs(MSK_SAVE_PATH, exist_ok=True)

def restrict_neighborhood(size_mask_neighborhood, h, w):
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
  
    for i in range(h):
        for j in range(w):
            for p in range(2 * size_mask_neighborhood + 1):
                for q in range(2 * size_mask_neighborhood + 1):
                    if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                        continue
                    if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

                    # print(i,j,p,q)
    mask = mask.reshape(h * w, h * w)

    return mask


from collections import defaultdict
shape2vns = defaultdict(list)
hw_list = []
vn2cnt = defaultdict(int)

for vn in sorted(os.listdir(VID_PATH)):

    for img_n in sorted(os.listdir(os.path.join(VID_PATH, vn)))[:1]:

        img_pth = os.path.join(VID_PATH, vn, img_n)
        img = Image.open(img_pth)

        _w, _h = img.size
        
        # if (_h, _w) not in shape2vns:
        shape2vns[(_h, _w)].append(vn)
        # else:
        #     shape2vns[(_h, _w)] += 1

        # make sure the image size can be divided by 64
        for rn in ROUND_NUMBER:

            # scaling 
            if scale_size is not None:
                if len(scale_size) == 1:
                    if(_w > _h):
                        _th = scale_size[0]
                        _tw = (_th * _w) / _h
                        _tw = int((_tw // rn) * rn)
                    else:
                        _tw = scale_size[0]
                        _th = (_tw * _h) / _w
                        _th = int((_th // rn) * rn) # make sure the image size can be divided by 64
                else:
                    _th = scale_size[1]
                    _tw = scale_size[0]

            # no scaling, just adjust with rn   
            else:
                _th = int((_h // rn) * rn)
                _tw = int((_w // rn) * rn ) 

              
            small_h, small_w = int(_th/PATCH_SIZE), int(_tw/PATCH_SIZE)
            hw_list.append((small_h, small_w))

            print('When round number is: {}, vn: {}, original_hw: {}, adjuedted_hw{}, small_hw: {}'.format(rn, vn, (_h, _w), (_th, _tw), (small_h, small_w)))
    
    for img_n in sorted(os.listdir(os.path.join(VID_PATH, vn))):
        vn2cnt[vn] += 1


shape2number = {}
for k, v in shape2vns.items():
    shape2number[k] = len(v)

print('Oirginal shape statics: ', shape2number)

for k,v in shape2vns.items():
    
    avg_length = 0
    for _v in v:
        avg_length += vn2cnt[_v] 
    avg_length /= len(v)

    print('Avearge length for shape {} videos is {}'.format(k, avg_length))

for size in [(1080,1920), (816,1920)]:
    print(size, shape2vns[size])


hw_list = list(set(hw_list))
print('Totally {} shapes, they are: {}'.format(len(hw_list), hw_list))

# hw_list += [(960//8,1440//8), (880//8,1320//8)]


# # Multi_thread using torch dataloader 
class _VideoGenerationDataset:
    """
    Dataset used to parallelize the reading of the timestamps
    of a list of videos, given their paths in the filesystem.
    Used in VideoClips and defined at top level so it can be
    pickled when forking.
    """

    def __init__(self, hw_list):
        
        self.hw_list = hw_list
        
    def __len__(self):
        return len(self.hw_list)

    def __getitem__(self, idx):
        
        h, w = self.hw_list[idx]

        print('Start generating {} r {}'.format((h,w), SIZE_MASK_NEIGHBORHOOD))
        mask = restrict_neighborhood(SIZE_MASK_NEIGHBORHOOD, h, w)  
        save_name = 'shape_{}_{}_r_{}.pth'.format(h, w, SIZE_MASK_NEIGHBORHOOD)

        print(os.path.join(MSK_SAVE_PATH, save_name))
        print('Saving {}'.format(save_name))

        torch.save(mask, os.path.join(MSK_SAVE_PATH, save_name))

        return mask

loader = torch.utils.data.DataLoader(
    _VideoGenerationDataset(hw_list),
    batch_size=1,
    num_workers=8,
)


for batch in tqdm(loader):
    print('one batch')
