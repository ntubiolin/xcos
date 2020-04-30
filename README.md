# xCos: An Explainable Cosine Metric for Face Verification Task

Official Pytorch implementation of "LxCos: An Explainable Cosine Metric for Face Verification Task" [arXiv](https://arxiv.org/abs/2003.05383)


<img src='https://imgur.com/v9SqRWd'>

<img src='./doc/gif_teaser.gif'>

<img src='./doc/gif_teaser2.gif'>

See YouTube [video demo](https://www.youtube.com/watch?v=87Vh1HDBjD0&list=PLPoVtv-xp_dL5uckIzz1PKwNjg1yI0I94&index=32&t=0s) or full resolution videos on [Google Drive](https://drive.google.com/open?id=1sT_ov_lLhQlXE2PnBKCUGOTNz5f_p9G7)

## Introduction

In "Free-form Video Inpainting with 3D Gated Convolution and Temporal PatchGAN. Chang et al. ICCV 2019.", we proposed 3D gated convolutions, Temporal PatchGAN and mask video generation algorithm to deal with free-form video inpainting in an end-to-end way. It is the first deep method for free-form video inpainting and achieves state-of-the-art performance both quantitatively and qualitatively. However, there are too many parameters for 3D gated convolutions and it takes long to train and inference.

Therefore, in "Learnable Gated Temporal Shift Module for Deep Video Inpainting. Chang et al. BMVC 2019.", we proposed a new LGTSM based on temporal shift module (TSM) for action recognition to reduce model parameters and training time to about 33%. The performance is almost the same as our previous work.

![lgtsm](doc/learnable_temporal_shift.png)

![block](doc/block_stacking.png)


This repository contains source code for both works. Some pretrained weights for the GTSM one are given, while the LGTSM code could be found in the LGTSM branch. The implementation of the baseline CombCN is also provided.

![compare](doc/fig_compare.png)
## Environment Setup
```
git clone git@github.com:amjltc295/Free-Form-Video-Inpainting.git
cd Free-Form-Video-Inpainting
git submodule update --init --recursive
conda env create -f environment.yaml
source activate free_form_video_inpainting
```

## Training
Please see [training](doc/training.md)

## Testing
1. Download corresponding pretrained weights from [Google Drive](https://drive.google.com/open?id=1uva9yI8yYKivqi4pWcyZLcCdIt1k-LRY)
    * The weights for the ICCV 2019 work
        * The one trained on FVI dataset is under `FFVI_3DGatedConv+TPGAN_trained_on_FVI_dataset` as well as its training config.
    * The weights for the BMVC 2019 work (LGTSM)
        * The one trained on FVI dataset is named as `v0.2.3_GatedTSM_inplace_noskip_b2_back_L1_vgg_style_TSMSNTPD128_1_1_10_1_VOR_allMasks_load135_e135_pdist0.1256`
        * For the one trained on FaceForensics, please refer to `Readme`

2. Update parameters in `src/other_configs/inference_example.json`:
    * If you want to test on other data, set `root_masks_dir` for testing masks and `root_videos_dir` for testing frames.
    * If you want to turn on evaluation, set `evaluate_score` to `true`.
3. Run
```
python train.py -r <pretrained_weight_path> --dataset_config other_configs/inference_example.json -od test_outputs
```

Then, you should have a directory src/test_outputs/ like:
```
test_outputs
└── epoch_0
    ├── test_object_like
    │   ├── inputs
    │   │   └── input_0000
    │   └── result_0000
    └── test_object_removal
        ├── inputs
        │   └── input_0000
        └── result_0000
```
The following GIFs show the figures that will appear in

(top row) `test_object_like/result_0000`, `test_object_like/inputs/result_0000`,

(bottom row) `test_object_removal/result_0000`, `test_object_removal/inputs/result_0000`

<img src='./doc/test_images/test_object_like.gif'>

<img src='./doc/test_images/test_object_removal.gif'>

## License
**This repository is limited to research purpose.** For any commercial usage, please contact us.

## Authors

Ya-Liang Chang (Allen) [amjltc295](https://github.com/amjltc295/) yaliangchang@cmlab.csie.ntu.edu.tw

Zhe-Yu Liu [Nash2325138](https://github.com/Nash2325138) zhe2325138@cmlab.csie.ntu.edu.tw


Please cite our papers if you use this repo in your research:
```
@article{chang2019free,
  title={Free-form Video Inpainting with 3D Gated Convolution and Temporal PatchGAN},
  author={Chang, Ya-Liang and Liu, Zhe Yu and Lee, Kuan-Ying and Hsu, Winston},
  journal={In Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2019}
}
@article{chang2019learnable,
  title={Learnable Gated Temporal Shift Module for Deep Video Inpainting"},
  author={Chang, Ya-Liang and Liu, Zhe Yu and Lee, Kuan-Ying and Hsu, Winston},
  journal={BMVC},
  year={2019}
}
```
## Acknowledgement
This work was supported in part by the Ministry of Science and Technology, Taiwan, under
Grant MOST 108-2634-F-002-004. We also benefit from the NVIDIA grants and the DGX-1
AI Supercomputer. We are grateful to the National Center for High-performance Computing.

# Pytorch Golden Template

## Class Diagram
<img src='./doc/PytorchTemplate-initialDesgin.png'>

## Features

## Setup

1. Use the template by clicking the green button or directly clone this repo
<img src='./doc/UseThisTemplate.png'>

2. Install miniconda/anaconda, a package for  package/environment management
```
wget repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

3. Build conda environment from file
```
cd pytorch-golden-template
conda create -n PytorchGoldenTemplate python">=3.6" pytorch">=1.0" torchvision tensorboard pillow">=6.1" pandas coloredlogs imageio scipy -c pytorch
source activate PytorchGoldenTemplate
pip install attrdict tensorboardX
# Or
conda env create -f environment.yaml
```

4. Update submodules
```
git submodule update --init --recursive
```

5. Activate the environment
```
source activate <environemnt name>
```

## Usage

```
usage: main.py [-h] [-tc TEMPLATE_CONFIG]
               [-sc SPECIFIED_CONFIGS [SPECIFIED_CONFIGS ...]] [-r RESUME]
               [-p PRETRAINED] [-d DEVICE] [--mode {train,test,eval}]
               [--saved_keys SAVED_KEYS [SAVED_KEYS ...]]
               [--ckpts_subdir CKPTS_SUBDIR] [--outputs_subdir OUTPUTS_SUBDIR]

PyTorch Template

optional arguments:
  -h, --help            show this help message and exit
  -tc TEMPLATE_CONFIG, --template_config TEMPLATE_CONFIG
                        Template configuraion file. It should contain all
                        default configuration and will be overwritten by
                        specified config.
  -sc SPECIFIED_CONFIGS [SPECIFIED_CONFIGS ...], --specified_configs SPECIFIED_CONFIGS [SPECIFIED_CONFIGS ...]
                        Specified configuraion files. They serve as experiemnt
                        controls and will overwrite template configs.
  -r RESUME, --resume RESUME
                        path to latest checkpoint (default: None)
  -p PRETRAINED, --pretrained PRETRAINED
                        path to pretrained checkpoint (default: None)
  -d DEVICE, --device DEVICE
                        indices of GPUs to enable (default: all)
  --mode {train,test,eval}
  --saved_keys SAVED_KEYS [SAVED_KEYS ...]
                        Specify the keys to save at testing mode.
  --ckpts_subdir CKPTS_SUBDIR
                        Subdir name for ckpts saving.
  --outputs_subdir OUTPUTS_SUBDIR
                        Subdir name for outputs saving.

```

Example: train a new model
```
python main.py
```

Example: train a new GAN model
```
python main.py -tc configs/template_GAN_train_config.json
```

Example: resume from a checkpoint, inference and save outputs
```
python main.py --resume ./saved/ckpts/template_config+CrossEntropy/0723_180600/ckpt-ep1-valid_mnist_avg_loss0.2885-best.pth --mode test
```

Example: generate testing results
```
# For submission to the leaderboard, etc.
python main.py --mode test -p <pretrained_weight>
```

Example: evaluate results
```
# You could evaluate results by other models (corresponding data loader needs to be defined)
# See the result_data_loaders in configs/template_eval_config.json for details
python main.py --mode eval
```

## Tensorboard Integration
By default, Tensorboard logs are recoded under saved/runs.

Please run ``` tensorboard --logidr saved/runs``` and go to `localhost:6006` to see the training progress on tensorboard.

## Folder Structure

## Authors
* Ya-Liang Chang (Allen) [amjltc295](https://github.com/amjltc295)
* Zhe Yu Liu [Nash2325138](https://github.com/Nash2325138)

