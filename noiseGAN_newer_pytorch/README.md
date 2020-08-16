# Synthesizing Camera Noise using Generative Adversarial Networks

## noiseGAN

We provide this implementation adapted to a newer PyTorch version.

This code is based on the [original CycleGAN implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/).

## Prerequisites
- Python 3
- NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation
Please, use a PyTorch version greater or equal to 0.4.1.

You can install the requirements by ```requirements.txt```:
```
pip install -r requirements.txt
```

**Note**: It seems the newest version of PyTorch+torchvision does not work well on Ubuntu 16.04+python3.5. Please install the versions inside ```requirements_torch1_14.txt```:

```
pip install -r requirements_torch1_14.txt
```

### Training a model
- Run ```python -m visdom.server``` in a console and enter http://localhost:8097 in order to view training results and loss plots.
- Run the train script:
```
python train.py --dataroot_A <path-to-lower-ISO> --dataroot_B <path-to-higher-ISO> --name <name-of-model> --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 40 --n_epochs_decay 0
```
For checking the parameters, please check the files in ```options``` folder.


### Testing our models
- Our trained models were trained in PyTorch 0.3.1. For newer versions, we are retraining the models. We provide one [trained model](https://drive.google.com/file/d/1INIqDRjVP1n0fvz8T8F55IvGc1znXv_G/). Please download and extract it to checkpoints, or run the following batch:
```bash
bash ./download_trained_models.sh
```

This model can be tested by:
```
python generate_noise.py --dataroot_A ./sample_imgs/SIDD_N_S6_clean --dataroot_B None --name SIDD_cleanTo3200_S6_new_pytorch --model test --dataset_mode single --no_dropout --crop_size 256
```

**Disclaimer:** The trained models provided for this implementation are not the same used for the manuscript reports. Please check the implementation using PyTorch 0.3.1 if you wish to replicate our results.