#########################################################
# Imports
#########################################################

######################################
# Misc
######################################
import os
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

######################################
# Change according to your needs
######################################
train_dataset_name = '/media/bernardo/Storage/dataset_iso/dataset_iso/train'
val_dataset_name = '/media/bernardo/Storage/dataset_iso/dataset_iso/val'
test_dataset_name = '/media/bernardo/Storage/dataset_iso/dataset_iso/test'

# Run parameters
batch_size = 32
epochs = 10

# Data specific constants
image_size = 128,128
classes = ['iso100','iso200','iso400','iso800','iso1600','iso3200']
num_classes = 6

model = create_model(image_size, num_classes=num_classes)
model.summary()


datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True)
datagen.config['random_crop_size'] = image_size
datagen.set_pipeline([random_transform,random_crop,random90rot,standardize,compute_fft2])
flow_train = datagen.flow_from_directory(train_dataset_name,batch_size=batch_size,color_mode='rgbfft',target_size=image_size)

datagen_val = ImageDataGenerator()
datagen_val.config['center_crop_size'] = image_size
datagen_val.set_pipeline([random_transform,center_crop,standardize,compute_fft2])
flow_val = datagen_val.flow_from_directory(val_dataset_name,batch_size=batch_size,color_mode='rgbfft',target_size=image_size)

filepath="trained_models/t3i_best_weights.h5"
callback_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights=True, mode='max')

# Training CNN
history = model.fit_generator(flow_train,
                    steps_per_epoch=200,
                    epochs=epochs,
                    verbose=1,
                    validation_data=flow_val,
                    validation_steps=500,
                    callbacks=[callback_checkpoint],
                    workers=8,use_multiprocessing=True)

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
