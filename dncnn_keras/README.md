# Synthesizing Camera Noise using Generative Adversarial Networks

## DnCNN Experiment

This directory provides code for the comparison of performance of the same denoiser (DnCNN-S) trained using different noise-models. This Keras implementation is based in the one provided in https://github.com/cszn/DnCNN. For more information of our experiment, please check the paper.

First, remember to install all the dependencies:
```
pip install -r requirements.txt
```

We provide the trained denoiser models for replicating the experiment. Please download the [trained models](https://drive.google.com/file/d/1Z33zkZkeDG4g94UHpid_u1RvXFiLGAzF) and extract them into ```trained_models```. You can run the following bash:
```bash
bash ./download_trained_models.sh
```

For replicating the experiment, please download the [RENOIR](http://ani.stat.fsu.edu/~abarbu/Renoir.html) and [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) datasets, and change the ```test.py``` script to account for the paths.