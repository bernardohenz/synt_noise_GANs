# Synthesizing Camera Noise using Generative Adversarial Networks

This repository provides implementations for generating synthetic camera-noise, as well as for the experiments for validating the proposed approach.

Check the README of each directory for more information about how to run them.


## Noise Synthesizer

[noiseGAN](noiseGAN) provides the code for synthesizing natural camera noise given the trained model. The code is based on the original [PyTorch implementation of CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/pytorch0.3.1), which used PyTorch 0.3.1. We provide the models trained with our Canon T3i dataset, and for the SIDD.

If you are having trouble installing PyTorch 0.3.1 (or just want to run it with a newer PyTorch version), we have provided an adaptation to our code to PyTorch versions greater than 0.3.1. Please check the [noiseGAN_newer_pytorch](noiseGAN_newer_pytorch) for more information.

## Validation Models

[noiseVal_artXnatural](noiseVal_artXnatural) provides the codes for the classifiers we trained for validating our method. It includes a binary classifier (artificial vs natural noise) and a multi-class classifier of several noise models, including ours. [noiseVal_isoRecognizer](noiseVal_isoRecognizer) provides the code for an ISO-level classifier trained in our T3i dataset.

## Quantitative Metrics

[noise_comparison_KL_KS](noise_comparison_KL_KS) provides the script for computing and comparing the Kullback-Leibler (KS) divergence and Kolmogorov-Smirnov (KS) of our method against existing ones. It also exports patches with these metrics for further visual inspection.

## Denoiser Application

[dncnn_keras](dncnn_keras) provides code for evaluating and comparing the training of the same denoiser when using distinct noise-models. The DnCNN trained using a combination of our noise-models is the one that achieved higher PSNR values in natural benchmarks.

## Citation
If you use this code, please cite our paper
```
@article{HenzGastalOliveira_2020,
    author = {Bernardo Henz and Eduardo S. L. Gastal and Manuel M. Oliveira},
    title   = {Synthesizing Camera Noise using Generative Adversarial Networks},
    journal = {IEEE Transactions on Visualization and Computer Graphics},
    volume = {},
    year    = {2020},
    doi     = {10.1109/TVCG.2020.3012120}
    }
```