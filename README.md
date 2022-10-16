# INO_VOS
The official code for [ACM MM 2022] 'In-N-Out Generative Learning for Dense Unsupervised Video Segmentation'.
[[arXiv]](https://arxiv.org/abs/2203.15312)





# Environment
We test with:
  * python==3.7
  * pytorch==1.7.1
  * CUDA==10.2
  
We train on Charades with 4x16GB V100 and Kinetics-400 with 8x16GB V100. The training takes around 12h and 1week, respectively. 
The codebase is implemented based on [DINO](https://github.com/facebookresearch/dino), [DUL](https://github.com/visinf/dense-ulearn-vos), and [VRW](https://github.com/ajabri/videowalk). 


# Data Preparation
We use [charades_480p](https://prior.allenai.org/projects/charades) and [Kinetics-400](https://github.com/cvdfoundation/kinetics-dataset) for trianing. We benchmark on DAVIS-2017 val and YouTube-VOS 2018 val.

After downloading datasets, run:
```shell
mkdir ./data
ln -s your/path/Charades_v1_480 ./data
ln -s your/path/Kinetics_400 ./data
```

The structure of ```data``` folder should be:
```shell
-data
  -Charades_v1_480
    - xxxx.mp4
    - ...
  -Kinetics_400
    - xxxx.mp4
    - ...
```

# Training

Set the ```ckpt_output_path```, ```dataset_cache_path``` in ```train_charades.sh``` as you need and then run 

```shell
  sh  train_charades.sh
```

The dataset meta will be cached under ```dataset_cache_path``` at the first run (it may take few minutes.).

Same for training on Kinetics-400.

# Evaluation 



# Citation
If you find our work useful, please consider citing:

```latex
@article{pan2022n,
  title={In-N-Out Generative Learning for Dense Unsupervised Video Segmentation},
  author={Pan, Xiao and Li, Peike and Yang, Zongxin and Zhou, Huiling and Zhou, Chang and Yang, Hongxia and Zhou, Jingren and Yang, Yi},
  journal={arXiv preprint arXiv:2203.15312},
  year={2022}
}
```

