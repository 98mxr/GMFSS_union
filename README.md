# GMFSS_union

High Performance GMFSS with RIFE and GAN for Video Frame Interpolation

---

* Cupy is required as a running environment, please follow the [link](https://docs.cupy.dev/en/stable/install.html) to install.
* The pre-trained model can be obtained from the following link, rename the folder to train_log and put it in the root directory, we provide two pre-trained models [prototype_vanillagan](https://drive.google.com/file/d/1AsA7a4HNR4RjCeEmNUJWy5kY3dBC-mru/view?usp=sharing) and [prototype_wgan](https://drive.google.com/file/d/1GAp9DljP1RCQXz0uu_GNn751NBMEQOUB/view?usp=sharing).

## Run Video Frame Interpolation

```
python3 inference_video.py --img=demo/ --scale=1.0 --multi=2
```

## Train

I can provide the training code of GMFSS_union, but it is really troublesome to organize the code. If anyone needs it, please contact me in issus.
