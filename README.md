# GMFSS_union

High Performance [GMFSS](https://github.com/YiWeiHuang-stack/GMFSS) with RIFE and GAN for Video Frame Interpolation

---

* Cupy is required as a running environment, please follow the [link](https://docs.cupy.dev/en/stable/install.html) to install.
* The pre-trained model can be obtained from the following link, rename the folder to train_log and put it in the root directory, we provide two pre-trained models [prototype_vanillagan](https://drive.google.com/file/d/1AsA7a4HNR4RjCeEmNUJWy5kY3dBC-mru/view?usp=sharing) and [prototype_wgan](https://drive.google.com/file/d/1GAp9DljP1RCQXz0uu_GNn751NBMEQOUB/view?usp=sharing).

## Run Video Frame Interpolation

```
python3 inference_video.py --img=demo/ --scale=1.0 --multi=2
```

## Train
2022.11.27, the training code has been made public. This code is run on a single V100, so there is no DistributedDataParallel related code. Please refer to [RIFE train](https://github.com/megvii-research/ECCV2022-RIFE/blob/main/train.py). The required training set is ATD-12K, which can be obtained from the [official library](https://drive.google.com/file/d/1XBDuiEgdd6c0S4OXLF4QvgSn_XNPwc-g/view). We also used a private data set, which unfortunately cannot be made public for the time being.In the process of organizing the code in the future, I found an error in the numerical processing of GAN Loss. I provided a replacement file, but the overall situation still maintained my original processing.

