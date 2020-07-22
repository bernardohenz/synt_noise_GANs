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
from util.extendable_datagen_several_classes import ImageDataGenerator, random_transform,standardize
from util.data_aug_transforms import random90rot,export_img,random_crop,center_crop,augment_contrast,augment_brightness,compute_fft2, srgb_to_linear
import numpy as np

import matplotlib.pyplot as plt

# We'll use matplotlib for graphics.
import matplotlib.patheffects as PathEffects
import matplotlib


# We import sklearn.
import sklearn
RS = 20150104
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.metrics import accuracy_score
from util.conf_matrix import plot_confusion_matrix
from model_creation import create_model
#########################################################
# General Run Parameters
#########################################################

######################################
# Change according to your needs
######################################

SELECTED_ISO = 800

test_clean_dir = '<path-to-SIDD-patches>/SIDD_{0:04d}/<clean-train>'.format(SELECTED_ISO)
test_noisy_dir = '<path-to-SIDD-patches>/SIDD_{0:04d}/<noisy-train>'.format(SELECTED_ISO)
test_GAN_dir = '<path-to-SIDD-patches>/SIDD_{0:04d}/<gan-train>'.format(SELECTED_ISO)

test_clean_dir = 'sample_imgs/SIDD_{0:04d}/clean'.format(SELECTED_ISO)
test_noisy_dir = 'sample_imgs/SIDD_{0:04d}/noisy'.format(SELECTED_ISO)
test_GAN_dir = 'sample_imgs/SIDD_{0:04d}/GAN'.format(SELECTED_ISO)


# Data specific constants
image_size = 128,128
#classes = ['gaussian','poisson','gaussianpoisson','mosaicgaussianpoisson','natural']
classes = ['Gaussian','Poisson','GaussPois','GaussMosaic','NoiseFlow','GAN_SIDD','Natural']
num_classes = 7


#load best model
datagenTest = ImageDataGenerator()
datagenTest.config['random_crop_size'] = image_size
datagenTest.set_pipeline([random_crop,standardize,compute_fft2])
flow_test = datagenTest.flow_from_directory(test_clean_dir,batch_size=50,color_mode='rgbfft',target_size=image_size)
flow_test.setCurrentISO(SELECTED_ISO, test_noisy_dir)
flow_test.setGANdir(test_GAN_dir)


flow_test.batch_size = 3#batchsizes_for_isos[str(ISO_LEVEL)]
total_batch_size = 500

x = np.zeros((total_batch_size,image_size[0],image_size[1],6))
y_true = np.zeros((total_batch_size,7))
iter_flow = iter(flow_test)
for i in range(total_batch_size//flow_test.batch_size):
    if (((i*flow_test.batch_size)%len(flow_test.filenames))==0):
        flow_test.on_epoch_end()
        iter_flow = iter(flow_test)
    x_cur,y_cur = next(iter_flow)
    x[i*flow_test.batch_size:(i+1)*flow_test.batch_size] = x_cur
    y_true[i*flow_test.batch_size:(i+1)*flow_test.batch_size]=y_cur

model = create_model(image_size, num_classes=num_classes)
model.load_weights('trained_models/{}_SIDD_several_classes_weights.h5'.format(SELECTED_ISO))

y_pred = model.predict(x)

y_true = y_true.argmax(axis=1)
y_pred = y_pred.argmax(axis=1)


plot_confusion_matrix(y_true,y_pred,  classes=classes, normalize=False,
                      title='Noise Type Classifier')

plt.savefig('conf_matrix_{}_sidd.pdf'.format(SELECTED_ISO))
newModel = Model(inputs=model.input, outputs=model.layers[-4].output )
x_feat_all = newModel.predict(x)

x_proj = TSNE(random_state=RS+2).fit_transform(x_feat_all)

from util.visualization import scatter

nb_points = 2000
flow_test.batch_size = nb_points


scatter(x_proj[:nb_points],y_true[:nb_points],labels=classes,chosen_palette='muted')
plt.savefig('{}_SIDD_w_legend_palette_muted.pdf'.format(SELECTED_ISO))
scatter(x_proj[:nb_points],y_true[:nb_points],labels=classes,legend=False,chosen_palette='muted')
plt.savefig('{}_SIDD_no_legend_palette_muted.pdf'.format(SELECTED_ISO))



