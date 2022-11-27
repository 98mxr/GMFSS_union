========================================================================================
=== Vimeo-90K Septuplet Dataset for Video Denoising, Deblocking, and Super-resolution
========================================================================================

Vimeo-90K septuplets dataset contains 91701 septuplets (each septuplets is a short RGB video sequence that consists of 7 frames) from 39k video clips with fixed resolution 448x256. This dataset is designed to video denoising, deblocking, and super-resolution. All the videos are downloaded from vimeo.com.

========================================================================================
=== Folder structure
========================================================================================

- sequences: This folder stores all 91701 septuplets. It uses a two-level folder structure, where each folder "%05d/%04d" contains a short video sequence consists of 7 frames: im1.png, im2.png, ..., im7.png.

- sep_trainlist.txt: contains the list of sequences for training.

- sep_testlist.txt: contains the list of sequences for test.

======================================================
=== Citation
======================================================

If you use this dataset in your work, please cite the following work:

@article{xue17toflow,
  author = {Xue, Tianfan and Chen, Baian and Wu, Jiajun and Wei, Donglai and Freeman, William T},
  title = {Video Enhancement with Task-Oriented Flow},
  journal = {arXiv},
  year = {2017}
}

For questions, please contact Tianfan Xue (tianfan.xue@gmail.com), Baian Chen(baian@mit.edu), Jiajun Wu (jiajunwu@mit.edu), or Donglai Wei(donglai@csail.mit.edu).

For more information, please refers to our project website and github repo:

Project website: http://toflow.csail.mit.edu/
Github repo: https://github.com/anchen1011/toflow

======================================================
=== Disclaimer
======================================================

This dataset is for non-commercial usage only.
