#########################################################
# Imports
#########################################################

######################################
# Misc
######################################
import os

######################################
# Keras
######################################
import keras
from keras.models import Model
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

######################################
# Change according to your needs
######################################
SELECTED_ISO = 3200 

train_clean_dir = '<path-to-SIDD-patches>/SIDD_patches_{0:04d}/<clean-train>'.format(SELECTED_ISO)
train_noisy_dir = '<path-to-SIDD-patches>/SIDD_patches_{0:04d}/<noisy-train>'.format(SELECTED_ISO)
val_clean_dir = '<path-to-SIDD-patches>/SIDD_patches_{0:04d}/<clean-val>'.format(SELECTED_ISO)
val_noisy_dir = '<path-to-SIDD-patches>/SIDD_patches_{0:04d}/<noisy-val>'.format(SELECTED_ISO)
test_dataset_name = '<path-to-SIDD-patches>/SIDD_patches_{0:04d}/<clean-test>'.format(SELECTED_ISO)
test_noisy_name = '<path-to-SIDD-patches>/SIDD_patches_{0:04d}/<noisy-test>'.format(SELECTED_ISO)

train_clean_dir = '/media/bernardo/Storage/SIDD_patches_{0:04d}/both_datasets/gts_train_classificator'.format(SELECTED_ISO)
train_noisy_dir = '/media/bernardo/Storage/SIDD_patches_{0:04d}/both_datasets/{0:04d}'.format(SELECTED_ISO,SELECTED_ISO)
val_clean_dir = '/media/bernardo/Storage/SIDD_patches_{0:04d}/both_datasets/gts_val_classificator'.format(SELECTED_ISO)
val_noisy_dir = '/media/bernardo/Storage/SIDD_patches_{0:04d}/both_datasets/{0:04d}'.format(SELECTED_ISO,SELECTED_ISO)
test_clean_dir = '/media/bernardo/Storage/SIDD_patches_{0:04d}/both_datasets/gts_val_classificator'.format(SELECTED_ISO)
test_noisy_dir = '/media/bernardo/Storage/SIDD_patches_{0:04d}/both_datasets/{0:04d}'.format(SELECTED_ISO,SELECTED_ISO)


# Run parameters
batch_size = 32
epochs = 3

# Data specific constants
image_size = 128,128

classes = ['artificial','natural']
num_classes = 1

model = create_model(image_size,num_classes)
model.summary()


datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True)

datagen.config['random_crop_size'] = image_size

datagen.set_pipeline([random_transform,random_crop,random90rot,standardize,compute_fft2])
flow_train = datagen.flow_from_directory(train_clean_dir,batch_size=batch_size,color_mode='rgbfft',target_size=image_size)
flow_train.setCurrentISO(SELECTED_ISO,train_noisy_dir)

datagen_val = ImageDataGenerator()

datagen_val.config['center_crop_size'] = image_size

datagen_val.set_pipeline([random_transform,center_crop,standardize,compute_fft2])
flow_val = datagen_val.flow_from_directory(val_clean_dir,batch_size=batch_size,color_mode='rgbfft',target_size=image_size)
flow_val.setCurrentISO(SELECTED_ISO,val_noisy_dir)

# Training CNN
history = model.fit_generator(flow_train,
                    steps_per_epoch=200,
                    epochs=epochs,
                    verbose=1,
                    validation_data=flow_val,
                    validation_steps=100,
                    workers=8,use_multiprocessing=True)

#load best model
#model = keras.models.load_model(filepath)
model.save('{}_SIDD_binary.h5'.format(SELECTED_ISO))
# Evaluation
datagen_test = ImageDataGenerator()


datagen_test.config['center_crop_size'] = image_size
datagen_test.set_pipeline([random_transform,center_crop,standardize, compute_fft2])


#evaluating testset
flow_test = datagen_test.flow_from_directory(test_clean_dir,batch_size=128,shuffle=False,color_mode='rgbfft',target_size=image_size)
flow_test.setCurrentISO(SELECTED_ISO,test_noisy_dir)
loss,acc = model.evaluate_generator(flow_test, workers=8,use_multiprocessing=False)

print("{} Accuracy: {:.2f}".format(SELECTED_ISO,acc))