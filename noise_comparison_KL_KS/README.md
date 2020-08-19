# Synthesizing Camera Noise using Generative Adversarial Networks

## Comparing using KL-divergence and KS Values

the ```kl_ks_per_cam.py``` script compares our method against other using the Kullback-Leibler (KS) divergence and Kolmogorov-Smirnov (KS) test.

Please install the requirements for the script:
```
pip install -r requirements.txt
```

We provided a sample directory (```sample_imgs_per_ISO_lighting_camera```) following the expected directory format. With this sample directory, it is possible to reproduce the comparison among patches (Figs 2, 10 and 11 from manuscript, and Figs 1-5 in the Appendix). Please run the the script:
```
python kl_ks_per_cam.py
```

For reproducing the KL and KS values from Table 1, you should download the data containing all image patches from ```small SIDD```, and adjust the path in the ```kl_ks_per_cam.py``` script.

If you want, you can download a [zip containing all patches](https://drive.google.com/file/d/1EYkYE3Wlkk79mmDXJL2NEPE4nd_e3qu_/view?usp=sharing) (size of 16GB), please extract inside the folder ```all_patches```. You can do that by running following batch:
```bash
bash ./download_all_patches.sh
```

For running the script in all patches, and also exporting the ones presented in the manuscript, run the following code:
```
python kl_ks_per_cam.py --imgs_dir ./data_per_ISO_lighting_camera --output_dir ./exports
```