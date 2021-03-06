#########################################################
# Imports
#########################################################

######################################
# Misc
######################################
import os
import argparse
DESIRED_LOG_LEVEL = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = DESIRED_LOG_LEVEL
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow.compat.v1 as tfv1
tfv1.logging.set_verbosity({
    '0': tfv1.logging.DEBUG,
    '1': tfv1.logging.INFO,
    '2': tfv1.logging.WARN,
    '3': tfv1.logging.ERROR
}.get(DESIRED_LOG_LEVEL))

######################################
# Keras
######################################
import keras
from keras.callbacks import ModelCheckpoint
######################################
# Utils
######################################
from util.extendable_datagen import ImageDataGenerator, random_transform,standardize
from util.data_aug_transforms import random90rot,export_img,random_crop,center_crop,augment_contrast,augment_brightness,compute_fft2, srgb_to_linear
from util.reports import plot_confusion_matrix, plot_confusion_matrix2
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model_creation import create_model

#########################################################
# General Run Parameters
#########################################################

parser = argparse.ArgumentParser(description='Test of artificial vs real vs GAN noise.')
parser.add_argument('--img_dir', type=str, default='./sample_imgs',
                help='Folder containing image patches of different ISO values.')
args = parser.parse_args()

######################################
# Change according to your needs
######################################
test_dataset_name = args.img_dir

# Run parameters
batch_size = 32

# Data specific constants
image_size = 128,128
classes = ['iso100','iso200','iso400','iso800','iso1600','iso3200']
num_classes = 6

model = create_model(image_size, num_classes=num_classes)
model.summary()

filepath="trained_models/t3i_best_weights.h5"

#load best model
model.load_weights(filepath)

# Evaluation
datagen_test = ImageDataGenerator()
datagen_test.config['center_crop_size'] = image_size
datagen_test.set_pipeline([random_transform,center_crop,standardize,compute_fft2])

#evaluating testset
flow_test = datagen_test.flow_from_directory(test_dataset_name,batch_size=128,shuffle=False,color_mode='rgbfft',target_size=image_size)
preds = model.predict_generator(flow_test, workers=8)
preds_max = preds.argmax(axis=1)

y_true = flow_test.classes
cnf_matrix=confusion_matrix(y_true,preds_max)
cnf_matrix = np.around(cnf_matrix,decimals=4)

plt.figure()
plot_confusion_matrix2(cnf_matrix, target_names=['ISO 100','ISO 200','ISO 400','ISO 800','ISO 1600','ISO 3200'], normalize=True,
                      title='Generated ISO -  Confusion matrix')
plt.savefig('generated_noise_conf_matrix_T3i.pdf')
