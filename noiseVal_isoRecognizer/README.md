# Synthesizing Camera Noise using Generative Adversarial Networks - Noise Classifiers

This is the implementation of an ISO-level classifier trained on our T3i dataset.

The result of this experiment can be seen in Fig 6 in the original manuscript.

## Requirements
For installing the requirements, just run:
```
pip install -r requirements.txt
```
We have the ```Keras``` and  ```Tensorflow``` versions fixed to the ones we've used. If you use different versions, you will probably need to adjust the code.


## Noise ISO level Classifier

File ```noise_isoRecognizer_T3i.py``` trains and evaluates a network for recognizing ISO level for our T3i dataset.

We can clearly see how easy it is for discriminating artificial from natural noise.

For testing our trained classifier, please download our [trained models](https://drive.google.com/file/d/149OEnQCBiMiSFKsjjf3rp7uazdZ9bUAU) (for this experiment), and extract them into ```trained_models```. You can do that by running the following bash:
```bash
bash ./download_trained_models.sh
```

By running the ```test_noise_isoRecognizer_T3i.py```, you can test the trained models in sample images (check ```sample_imgs``` folder):
```
python test_noise_isoRecognizer_T3i.py
```

You can download set with [more sample images] (https://drive.google.com/file/d/17_EErt51NOrOhS15fRID3R9HVM2F6lUH/view?usp=sharing) (~300MB), extracting them into ```many_sample_imgs``` directory. You can do that by running the following bash:
```bash
bash ./download_more_sample_imgs.sh
```

For running the test script on these images, you can use the ```img_dir``` parameter:
```
python test_noise_isoRecognizer_T3i.py --img_dir ./many_sample_imgs
```