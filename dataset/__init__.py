import os 
import time
import torch
from torchvision.datasets.samplers.clip_sampler import RandomClipSampler, UniformClipSampler, DistributedSampler
import hashlib
from dataset.Kinetics400 import Kinetics400
from augs import DataAugmentationDINO
import utils

def make_dataloader_vrw(args, epoch):
    # We employ the same dataloader as VRW.
    # Return the prepared dataloader.

    def _get_cache_path(filepath):

        h = hashlib.sha1(filepath.encode()).hexdigest()
        cache_path = os.path.join(filepath, h[:10] + ".pt")
        # cache_path = os.path.join(filepath, 'e3ff6593ed.pt')
        cache_path = os.path.expanduser(cache_path)
        return cache_path

    def make_dataset(args, transform=None, cached=None):
        
        return Kinetics400(args.csv_path,
                frames_per_clip=args.seq_length, # 4
                step_between_clips=1,
                transform=transform, 
                extensions=('mp4'),
                frame_rate=args.frame_skip, # 8
                # cached=cached,
                _precomputed_metadata=cached)

    st = time.time()
    cache_path = _get_cache_path(args.cache_path)

    # make transform
    transform_train = DataAugmentationDINO(
            args, 
            args.global_crops_scale,
            args.local_crops_scale,
            args.local_crops_number,
        )
    
    # make dataset
    if args.cache_path and os.path.exists(cache_path):
        # load from cached metadata
        print("Loading dataset_train from {}".format(cache_path))
        dataset = torch.load(cache_path)
        cached = dict(video_paths=dataset.video_clips.video_paths,
                video_fps=dataset.video_clips.video_fps,
                video_pts=dataset.video_clips.video_pts)
        dataset = make_dataset(args, transform=transform_train, cached=cached)
    else:
        # generate metadata
        dataset = make_dataset(args, transform=transform_train)
        print("Saving dataset_train to {}".format(cache_path))
        utils.mkdir(os.path.dirname(cache_path))
        dataset.transform = None
        torch.save(dataset, cache_path)
        dataset.transform = transform_train
    
    # see VideoClips class in https://github.com/pytorch/vision/blob/main/torchvision/datasets/video_utils.py
    dataset.video_clips.compute_clips(args.seq_length, 1, frame_rate=args.frame_skip)
    print("Took", time.time() - st)

    # make sampler
    def make_data_sampler(is_train, dataset):
        torch.manual_seed(0)
        if hasattr(dataset, 'video_clips'):
            _sampler = RandomClipSampler #UniformClipSampler
            if args.distributed:
                print('!!! Using distributed sampler ... ')
                return DistributedSampler(_sampler(dataset.video_clips, args.clips_per_video))
            else:
                print('!!! Not using distributed smapler ... ')
                return _sampler(dataset.video_clips, args.clips_per_video)
        else:
            return torch.utils.data.sampler.RandomSampler(dataset) if is_train else None
    train_sampler = make_data_sampler(True, dataset)

    # make dataloader
    print("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.batch_size_per_gpu, # shuffle=not args.fast_test,
                    sampler=train_sampler, 
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True)

    return data_loader