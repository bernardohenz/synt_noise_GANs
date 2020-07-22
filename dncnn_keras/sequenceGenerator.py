import numpy as np
import re
from six.moves import range
import os,random
import sys
import threading
import copy
import inspect
import types
import keras
import glob, math, random
from keras import backend as K
import imageio
from data_aug import random_crop,random90rot

from skimage.util.noise import random_noise

def generateGaussianRandomVarNoiseChannelSpecific(img):
    out = np.zeros_like(img)
    for channel in range(img.shape[-1]):
      choosen_var = np.random.uniform(0.0001,0.0025)# sigma from 0.01~0.05
      out[:,:,channel] = random_noise(img[:,:,channel],mode='gaussian',var=choosen_var)
    return out
    
def generatePoissonNoise(img):
    return random_noise(img,mode='poisson')

def generateGaussianPoissonNoise(img):
    gauss_out = generateGaussianRandomVarNoiseChannelSpecific(img)
    poiss_out = generatePoissonNoise(gauss_out)
    return poiss_out

def generateGaussianRandomVarNoise(img):
    #choosen_var = np.random.uniform(0.0005,0.01) # sigma from 0.02~0.1
    choosen_var = np.random.uniform(0.0001,0.0025) # sigma from 0.01~0.05
    #sigma = 10**(np.random.uniform(-4,-1))
    #sigma = sigma**2

    return np.clip(random_noise(img, mode='gaussian', var= choosen_var ),0,1)




class Generator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, batch_size=32, crop_img_size=(256,256), data_dir=None,shuffle=True, seed=None):

        seed = seed or 13

        self.data_dir = data_dir

        if os.path.exists(os.path.join(data_dir,'filelist.txt')):
            with open(os.path.join(data_dir,'filelist.txt'),'r') as f:
                self.filenames_inputs = f.readlines()
            # remove whitespace characters like `\n` at the end of each line
            self.filenames_inputs = [x.strip() for x in self.filenames_inputs]
        else:
            self.filenames_inputs = glob.glob(os.path.join(data_dir,'**/*.png'),recursive=True)

        #self.list_noisy_dirs = ['/media/bernardo/Storage/cbd_net_demosaic_dataset/train']
        #self.list_noisy_dirs = ['/media/bernardo/Storage4T2/demosaic_dataset1600/images/train','/media/bernardo/Storage4T2/demosaic_dataset800/images/train','/media/bernardo/Storage4T2/demosaic_dataset400/images/train','/media/bernardo/Storage4T2/demosaic_dataset3200/images/train']
        self.list_noisy_dirs = ['/mnt/69aabd34-7b15-4a17-899c-9005f51d5076/demosaic_dataset_SIDD_several_cams/images/train']#['/media/bernardo/FasterDestroyer/demosaic_dataset3200/images/train','/media/bernardo/FasterDestroyer/demosaic_dataset1600/images/train','/media/bernardo/Storage/demosaic_dataset1600_mi3/images/train','/media/bernardo/Storage/demosaic_dataset1600_s90/images/train']
        self.nb_sample = len(self.filenames_inputs)
        self.batch_size = batch_size
        self.crop_img_size = crop_img_size
        self.shuffle = shuffle
        self.seed = seed
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        lenn = int(np.floor(self.nb_sample) / self.batch_size)
        if (self.nb_sample % self.batch_size >0):
            lenn +=1
            #print('Remainder: {}'.format(self.nb_sample % self.batch_size))
        return lenn

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        if (  (index+1)*self.batch_size <= len(self.indexes)):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:]
        # Find list of IDs
        #print('Indexes: {}'.format(indexes))
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filenames_inputs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_indexes):
        'Generates data containing batch_size samples'
        # Initialization
        batch_x = np.zeros((len(list_indexes), self.crop_img_size[0],self.crop_img_size[1], 3))

        batch_y = np.zeros((len(list_indexes), self.crop_img_size[0],self.crop_img_size[1], 3))
        # Generate data
        for i, cur_index in enumerate(list_indexes):
            fname = os.path.join(self.data_dir,self.filenames_inputs[cur_index])
            rgb_output = imageio.imread(fname).astype('float32') / 255.0

            #only gaussian noise
            #rgb_linear = rgb_output**2.2  ##sRGB to linear
            #rgb_input = generateGaussianPoissonNoise(rgb_output)
            #rgb_input = rgb_input**(1/2.2) ##linear to sRGB

            choosen_dir = random.choice(self.list_noisy_dirs)
            if ('9999' in self.filenames_inputs[cur_index]):
                rgb_input = rgb_output
            else:
                fnamein = os.path.join(choosen_dir,self.filenames_inputs[cur_index]).replace('.png','_fake_B.png')
                rgb_input = imageio.imread(fnamein).astype('float32') / 255.0
                rand_alpha = np.random.uniform(0.7,1.0)
                rgb_input = rand_alpha*rgb_input + (1-rand_alpha)*rgb_output

            xx,yy = random_crop(rgb_input,rgb_output,self.crop_img_size)
            xx,yy = random90rot(xx,yy)
            batch_x[i] = xx
            batch_y[i] = yy
            # build batch of labels
        return batch_x, batch_y
