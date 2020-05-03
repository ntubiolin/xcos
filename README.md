# xCos: An Explainable Cosine Metric for Face Verification Task

Official Pytorch implementation of "xCos: An Explainable Cosine Metric for Face Verification Task" [arXiv](https://arxiv.org/abs/2003.05383)


## Introduction
In "xCos: An Explainable Cosine Metric for Face Verification Task", we propose a novel similarity metric, called explainable cosine xCos, that comes with a learnable module that can be plugged into most of the verification models to provide meaningful explanations.
<img src='./doc/idea.png'>
State-of-the-art face verification models extract deep features of a pair of face images and compute the cosine similarity or the L2-distance of the paired features. Two images are said to be from the same person if the similarity is larger than a threshold value. However, with this standard procedure, we can hardly interpret these high dimensional features with our knowledge.
<img src='./doc/compare_xcos_with_grad_cam.png'>
Although there are some previous works attempting to visualize the results on the input images with saliency map, these saliency map based visualizations are mostly used for the localization of objects in a single image rather the similarity of two faces. Therefore, our framework provides a new verification branch to calculate similarity maps and discriminative location maps based on the features extracted from two faces. This way, we can strike a balance between verification accuracy and visual interpretability.
<img src='./doc/architecture.png'>

## Environment Setup
```
git clone git@github.com:ntubiolin/xcos.git
cd xcos
conda env create -f environment_xcos_template.yml
source activate xcos_template
```

## Training/ Testing
Please refer to the [pytorch-golden-template](https://github.com/amjltc295/pytorch-golden-template) for detailed config options.

## License
**This repository is limited to research purpose.** For any commercial usage, please contact us.

## Authors
Yu-sheng Lin [ntubiolin](https://github.com/ntubiolin) biolin@cmlab.csie.ntu.edu.tw

Zhe-Yu Liu [Nash2325138](https://github.com/Nash2325138) zhe2325138@cmlab.csie.ntu.edu.tw

Yuan Chen

Yu-Siang Wang

Hsin-Ying Lee

Yirong Chen

Ya-Liang Chang (Allen) [amjltc295](https://github.com/amjltc295/) yaliangchang@cmlab.csie.ntu.edu.tw

Please cite our papers if you use this repo in your research:
```
@article{Lin2020xCosAE,
  title={xCos: An Explainable Cosine Metric for Face Verification Task},
  author={Yu-sheng Lin and Zheyu Liu and Yuan Chen and Yu-Siang Wang and Hsin-Ying Lee and Yirong Chen and Ya-Liang Chang and Winston H. Hsu},
  journal={ArXiv},
  year={2020},
  volume={abs/2003.05383}
}
```
## Acknowledgement
This work was supported in part by the Ministry of Science and Technology, Taiwan, under Grant MOST 109-2634-F-002-032. We benefit from the NVIDIA grants and the DGX-1 AI Supercomputer and are also grateful to the National Center for High-performance Computing.