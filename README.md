# INO_VOS
The official code for **[ACM MM 2022] 'In-N-Out Generative Learning for Dense Unsupervised Video Segmentation'**.
[[arXiv]](https://arxiv.org/abs/2203.15312)

We achieve a new state-of-the-art performance for unsupervised learning methods on VOS task, based on ViT and the idea of generative learning. 

![image](https://user-images.githubusercontent.com/47111102/196872637-17c187f2-a2d3-4468-a23a-75be56bd48bd.png)
![image](https://user-images.githubusercontent.com/47111102/196040697-ca426c98-d3a4-4499-a9c7-54173a575fa9.png)
![image](https://user-images.githubusercontent.com/47111102/196040701-ea9e09f3-319e-4504-ab2a-5060a82edfee.png)


# Environment
We test with:
  * python==3.7
  * pytorch==1.7.1
  * CUDA==10.2
  
We train on Charades with 4x16GB V100 and Kinetics-400 with 8x16GB V100. The training takes around 12h and 1week, respectively. 
The codebase is implemented based on [DINO](https://github.com/facebookresearch/dino), [DUL](https://github.com/visinf/dense-ulearn-vos), and [VRW](https://github.com/ajabri/videowalk). 


# Dataset Preparation

### Training Datasets
We use [charades_480p](https://prior.allenai.org/projects/charades) and [Kinetics-400](https://github.com/cvdfoundation/kinetics-dataset) for training.

After downloading datasets, run:
```shell
git clone git@github.com:pansanity666/INO_VOS.git
cd INO_VOS
mkdir ./data
ln -s /your/path/Charades_v1_480 ./data
ln -s /your/path/Kinetics_400 ./data
```

### Evaluation Datasets
We benchmark on DAVIS-2017 val and YouTube-VOS 2018 val.

Download DAVIS-2017 from [here](https://github.com/davisvideochallenge/davis-2017/blob/master/data/get_davis.sh).
<!--  ```shell 
 cd $DAVIS_SAVE_DIR
 git clone https://github.com/davisvideochallenge/davis-2017 && cd davis-2017
 ./data/get_davis.sh
 cd $HOME
 ln -s $DAVIS_SAVE_DIR/davis-2017/DAVIS ./data
 ``` -->


Download YouTube-VOS 2018 (valid_all_frames.zip and valid.zip) from [here](https://competitions.codalab.org/competitions/19544#participate-get-data).

Link them to ```./data``` (similar as previous datasets).

 
The final structure of ```data``` folder should be:
```shell
-data
  -Charades_v1_480
    - xxxx.mp4
    - ...
  -Kinetics_400
    - xxxx.mp4
    - ...
  -DAVIS
    - Annotations
    - JPEGImages
    - ...
  -YouTube_VOS
    - valid_all_frames
    - valid
```

# Training

Set the ```ckpt_output_path``` in ```train_charades.sh``` as you need and then run 

```shell
# under INO_VOS dir
sh  train_charades.sh
```

The dataset meta will be cached under ```./cached/charades``` at the first run (it may take few minutes.).

Same for training on Kinetics-400.

# Evaluation 

### Inference
Our checkpoint used in the paper can be downloaded from [here](https://drive.google.com/drive/folders/1gf5XZ8Y9OPhXcsgzlI3Cp3h3dGQOm8My?usp=sharing).

For the sake of efficiency, we first pre-generate the neighbor masks used during label propagation and cache them on disk. 

```shell
python ./scripts/pre_calc_maskNeighborhood.py [davis|ytvos] 
```

It may take few minutes, and the neighbor masks will be cached under ```./cached/masks``` by default. 

Then, run label propagation via:

```shell
sh infer_vos.sh [davis|ytvos] $CKPT_PATH 
```

Two folders will be created under ```./results```, where ```vos``` is the segmentation masks while ```vis``` is the blended visualization results. 

### Evaluation: DAVIS-2017

 
 Please install the official evaluation code and evaluate the inference results:
 ```shell
# under INO_VOS dir
git clone https://github.com/davisvideochallenge/davis2017-evaluation ./davis2017-evaluation
python ./davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path $OUTPUT_VOS --davis_path ./data/DAVIS/ 
 ```
 
 
 
 
### Evaluation: YouTube-VOS 2018

Please use the official [CodaLab evaluation server](https://competitions.codalab.org/competitions/19544#participate-submit_results).
To create the submission, rename the `vos`-directory to `Annotations` and compress it to `Annotations.zip` for uploading.
 



# Citation
If you find our work useful, please consider citing:

```latex
@inproceedings{pan2022n,
  title={In-n-out generative learning for dense unsupervised video segmentation},
  author={Pan, Xiao and Li, Peike and Yang, Zongxin and Zhou, Huiling and Zhou, Chang and Yang, Hongxia and Zhou, Jingren and Yang, Yi},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={1819--1827},
  year={2022}
}
```

