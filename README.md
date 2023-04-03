# GMFSS_union

High Performance [GMFSS](https://github.com/YiWeiHuang-stack/GMFSS) with RIFE and GAN for Video Frame Interpolation

---

**2023-04-03: We now provide [GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna) as a factual basis for training in GMFSS. Please use it. This item will not be updated!**

---

* Cupy is required as a running environment, please follow the [link](https://docs.cupy.dev/en/stable/install.html) to install.
* The pre-trained model can be obtained from the following link, rename the folder to train_log and put it in the root directory, we provide two pre-trained models [prototype_vanillagan](https://drive.google.com/file/d/1AsA7a4HNR4RjCeEmNUJWy5kY3dBC-mru/view?usp=sharing) and [prototype_wgan](https://drive.google.com/file/d/1GAp9DljP1RCQXz0uu_GNn751NBMEQOUB/view?usp=sharing).

## Run Video Frame Interpolation

```
python3 inference_video.py --img=demo/ --scale=1.0 --multi=2
```

## Train
2023.02.10，The training code has been re-released, although nothing has actually changed. The official training log and results can be found in [Google Drive](https://drive.google.com/file/d/1s5uS-psfn61-22GRY4wlAE037xlJSzpm/view?usp=share_link). Please note that this result is only used to verify the training process, for inference please still use the version released above. Finally I deleted the RIFE_Fix_GAN_Loss_output.py, which only increases the misunderstanding

2022.11.27, the training code has been made public. This code is run on a single V100, so there is no DistributedDataParallel related code. Please refer to [RIFE train](https://github.com/megvii-research/ECCV2022-RIFE/blob/main/train.py). The required training set is ATD-12K, which can be obtained from the [official library](https://drive.google.com/file/d/1XBDuiEgdd6c0S4OXLF4QvgSn_XNPwc-g/view). We also used a private data set, which unfortunately cannot be made public for the time being.In the process of organizing the code in the future, I found an error in the numerical processing of GAN Loss. I provided a replacement file, but the overall situation still maintained my original processing.


## Acknowledgment
This project is supported by [SVFI](https://steamcommunity.com/app/1692080) [Development Team](https://github.com/Justin62628/Squirrel-RIFE) 
