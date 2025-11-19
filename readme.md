# [Under the Shadow: Exploiting Opacity Variation for Fine-grained Shadow Detection](https://neurips.cc/virtual/2025/loc/san-diego/poster/118471)

Official implementation of **AdaVec**, from the following paper:

Less is More: Efficient Image Vectorization with Adaptive Parameterization, CVPR 2025

[[`Paper`](https://tangyuan-x.github.io/UndertheShadow_website/static/pdfs/paper.pdf)] [[`Video`](https://tangyuan-x.github.io/UndertheShadow_website/)] [[`Project`](https://tangyuan-x.github.io/UndertheShadow_website/)] [[`BibTeX`](#Reference)]

![![title]](imgs/network-4.png?raw=true)

## Installation
We suggest users to use the conda for creating new python environment. 

```bash
git clone https://github.com/IMU-Group/FGSD.git
cd FGSD
conda create -n FGSD python=3.8
conda activate FGSD
pip install -r requirements.txt

```

## Run Experiments 
```bash
conda activate FGSD
cd FGSD
python train.py 
```

## Reference

    @inproceedings{Qiao2025Under,
      title={Under the Shadow: Exploiting Opacity Variation for Fine-grained Shadow Detection},
      author={Qiao, Xiaotian and Xu, Ke  and Yang, Xianglong and Dong, Ruijie and Xia, Xiaofang and Cui, Jiangtao },
      booktitle={Neural Information Processing Systems},
      year={2025},
    }

## Acknowledgement
Our implementation is mainly based on the [FDRNet](https://github.com/rayleizhu/FDRNet). We gratefully thank the authors for their wonderful works.
