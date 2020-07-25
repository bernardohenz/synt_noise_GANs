# Synthesizing Camera Noise using Generative Adversarial Networks - Noise Classifiers

This is the implementation of our noise classifiers. 

For more information, please check our paper in Section 6.

Our experiments were done over sRGB patches from the [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/) (small version).
In our experiments, the SIDD images were first split into ISO values, then train/val/test, following by cropping into 256x256 patches.
For using the scripts, the code must be changed to account for the directories of the patches.

## Requirements
For installing the requirements, just run:
```
pip install -r requirements.txt
```
We have the ```Keras``` and  ```Tensorflow``` versions fixed to the ones we've used. If you use different versions, you will probably need to adjust the code.


## Common Noise Models vs Natural Noise

File ```noise_artXreal.py``` trains and evaluates a network for discriminating artificial noise from natural noise.

This script compares existing models (Gaussian, Poisson, GaussianPoissonian, and Gaussian through CFA+demosaicing) against 
natural noise.

We can clearly see how easy it is for discriminating artificial from natural noise.

## Classification of Several Noise Models

File ```noise_artXrealXGAN_SIDDXnoiseflow.py``` trains a classifier for discriminating the above mentioned methods, besides NoiseFlow and ours.

If you wish to download our [trained models](https://drive.google.com/file/d/149OEnQCBiMiSFKsjjf3rp7uazdZ9bUAU) (for this experiment), and extract them into ```trained_models```. You can do that by running the following bash:
```bash
bash ./download_trained_models.sh
```

By running the ```test_artXrealXGAN_SIDDXnoise_flow```, you can test the trained models in sample images (check ```sample_imgs``` folder).
