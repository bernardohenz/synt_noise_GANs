#########################################################
# Imports
#########################################################

######################################
# Misc
######################################
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
######################################
# Keras
######################################
import keras
from keras.models import Model
######################################
# Utils
######################################
from util.extendable_datagen_several_classes import ImageDataGenerator, random_transform,standardize
from util.data_aug_transforms import random90rot,export_img,random_crop,center_crop,augment_contrast,augment_brightness,compute_fft2, srgb_to_linear
from util.reports import plot_confusion_matrix, plot_confusion_matrix2
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model_creation import create_model
#########################################################
# General Run Parameters
#########################################################


######################################
# Change according to your needs
######################################
SELECTED_ISO = 3200 

train_clean_dir = '<path-to-SIDD-patches>/SIDD_{0:04d}/<clean-train>'.format(SELECTED_ISO)
train_noisy_dir = '<path-to-SIDD-patches>/SIDD_{0:04d}/<noisy-train>'.format(SELECTED_ISO)
train_GAN_dir = '<path-to-SIDD-patches>/SIDD_{0:04d}/<gan-train>'.format(SELECTED_ISO)

#Take as reference the 'sample_imgs' directory

train_clean_dir = '/media/bernardo/Storage/SIDD_patches_{0:04d}/both_datasets/gts_train_classificator'.format(SELECTED_ISO)
train_noisy_dir = '/media/bernardo/Storage/SIDD_patches_{0:04d}/both_datasets/{0:04d}'.format(SELECTED_ISO,SELECTED_ISO)
train_GAN_dir = '/media/bernardo/Storage/SIDD_patches_{0:04d}/both_datasets/GAN_fakes_{0:04d}'.format(SELECTED_ISO,SELECTED_ISO)



# Run parameters
batch_size = 32
epochs = 7

# Data specific constants
image_size = 128,128
#classes = ['gaussian','poisson','gaussianpoisson','mosaicgaussianpoisson','noiseflow','GAN','natural']
num_classes = 7

model = create_model(image_size, num_classes)
model.summary()


datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True)

datagen.config['random_crop_size'] = image_size

datagen.set_pipeline([random_transform,random_crop,random90rot,standardize,compute_fft2])
flow_train = datagen.flow_from_directory(train_clean_dir,batch_size=batch_size,color_mode='rgbfft',target_size=image_size)
flow_train.setCurrentISO(SELECTED_ISO, train_noisy_dir)
flow_train.setGANdir(train_GAN_dir)

# Training CNN
history = model.fit_generator(flow_train,
                    steps_per_epoch=1000,
                    epochs=epochs,
                    verbose=1,
                    workers=8)

model.save('{}_SIDD_several_classes_h5_7_epochs.h5'.format(SELECTED_ISO))

