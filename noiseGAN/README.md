# Synthesizing Camera Noise using Generative Adversarial Networks

## noiseGAN

We provide a PyTorch implementations for paper entitled "Synthesizing Camera Noise using Generative Adversarial Networks".

This implementation is based on the [original PyTorch implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/pytorch0.3.1) of the CycleGAN. We used an old branch of this repository. For more information, check below.

## Prerequisites
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation
We used PyTorch 0.3.1 and CUDA 10.0.
- Install the requirements inside ```requirements.txt```:
```
pip install -r requirements.txt
```


### Training a model
- Run ```python -m visdom.server``` in a console and enter http://localhost:8097 in order to view training results and loss plots.
- Run the train script:
```
python train.py --dataroot_A <path-to-lower-ISO> --dataroot_B <path-to-higher-ISO> --name <name-of-model> --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --niter 40 --niter_decay 0
```
For checking the parameters, please check the files in ```options``` folder.


### Downloading and testing our models
- Download our [trained models](https://drive.google.com/file/d/1uiPvJvqACe6WFTzY8xRftvmErnR44U1J) (~500MB) and extract them inside ```checkpoints```. You can run the following bash:
```bash
bash ./download_trained_models.sh
```
- Run the test script
```
python generate_noise.py --dataroot_A <input-imgs-dir> --dataroot_B None --name <name-of-model> --model test --dataset_mode single --no_dropout
```
- By using the following command, you can generate noise for 3 sample images
```
python generate_noise.py --dataroot_A ./sample_imgs/SIDD_N_S6_clean --dataroot_B None --name sidd_cleanTo3200_S6 --model test --dataset_mode single --no_dropout
```
