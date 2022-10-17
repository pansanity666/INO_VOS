import torchvision.datasets.video_utils

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset

import numpy as np

# NOTE: We use Kinetics400 dataset class for all the training dataset (Charades, Kinetics, YT-VOS).
class Kinetics400(VisionDataset):
    """
    `Kinetics-400 <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>`_
    dataset.
    Kinetics-400 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.
    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.
    Internally, it uses a VideoClips object to handle clip creation.
    Args:
        root (string): Root directory of the Kinetics-400 Dataset.
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.
    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

    def __init__(self, csv_path, frames_per_clip, step_between_clips=1, frame_rate=None,
                 extensions=('mp4',), transform=None, cached=None, _precomputed_metadata=None):
        super(Kinetics400, self).__init__(csv_path)
        extensions = extensions
        filenames = open(csv_path).readlines()
        video_list = [filename.split(',')[0].strip() for filename in filenames] # [v1name, v2name, ...]
    
        # input: video_list
        # https://github.com/pytorch/vision/blob/main/torchvision/datasets/video_utils.py
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata, # precomputed metadata
            num_workers=64
        )

        self.transform = transform

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):

        success = False
        while not success:
            try:
                video, audio, info, _ = self.video_clips.get_clip(idx)
                success = True
                
            # print('skippe`d idx', idx)
            # print(self.video_clips.num_clips(), idx)
                    
            except:
                print('skippe`d idx', idx)
                print(self.video_clips.num_clips(), idx)
                assert False
                idx = np.random.randint(self.__len__())

        # label = self.samples[video_idx][1]
        if self.transform is not None:
            video = [self.transform(v) for v in video]

        return video
    
