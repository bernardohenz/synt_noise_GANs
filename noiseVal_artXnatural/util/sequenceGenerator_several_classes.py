import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import sys
import threading
import copy
import inspect
import types
import keras
import random
import imageio
import pandas as pd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras import backend as K
from util.noise_utils import generateGaussianRandomVarNoiseChannelSpecific, generatePoissonNoise, generateGaussianPoissonNoise, generateDemosaicWithGaussianPoissonNoise
from util.data_aug_transforms import srgb_to_linear


# Values of std per ISO -> for Gaussian noise
gaussian_stds = {}
gaussian_stds['400'] = {}
gaussian_stds['400']['L'] = 0.04818
gaussian_stds['400']['N'] = 0.03685
gaussian_stds['400']['both_dataset'] = 0.042003
gaussian_stds['800'] = {}
gaussian_stds['800']['L'] = 0.0592269
gaussian_stds['800']['N'] = 0.0457938
gaussian_stds['800']['both_dataset'] = 0.053584
gaussian_stds['1600'] = {}
gaussian_stds['1600']['L'] = 0.0906989
gaussian_stds['1600']['N'] = 0.05532882
gaussian_stds['1600']['both_dataset'] = 0.07138
gaussian_stds['3200'] = {}
gaussian_stds['3200']['L'] = 0.139323
gaussian_stds['3200']['N'] = 0.12520
gaussian_stds['3200']['both_dataset'] = 0.13524


# Line configs of noiseflow for ISOs
NF_cam_isos = {}
NF_cam_isos['400'] = [1,7]
NF_cam_isos['800'] = [2,5,8,-1,-3]
NF_cam_isos['1600'] = [3,9]
NF_cam_isos['3200'] = [-5]


from noiseflow_code.borealisflows.NoiseFlowWrapper import NoiseFlowWrapper

patch_size, stride = 32, 32  # patch size  = [32, 32, 4]
aug_times = 1
scales = [1]  # [1, 0.9, 0.8, 0.7]
nf_model_path = 'noiseflow_code/models/NoiseFlow'

def load_cam_iso_nlf():
    cin = pd.read_csv('noiseflow_code/cam_iso_nlf.txt')
    cin = cin.drop_duplicates()
    cin = cin.set_index('cam_iso', drop=False)
    return cin

# Prepare NoiseFlow
noise_flow = NoiseFlowWrapper(nf_model_path)

# camera IDs and ISO levels related to the SIDD dataset
cam_iso_nlf = load_cam_iso_nlf()
n_cam_iso = cam_iso_nlf['cam_iso'].count()
iso_vals = [100.0, 400.0, 800.0, 1600.0, 3200.0]
cam_ids = [0, 1, 3, 3, 4]  # IP, GP, S6, N6, G4
cam_vals = ['IP', 'GP', 'S6', 'N6', 'G4']


def pack_raw(rgb_img):
    """Packs Bayer image to 4 channels (h, w) --> (h/2, w/2, 4)."""
    # pack Bayer image to 4 channels
    im = np.expand_dims(rgb_img, axis=2)
    img_shape = im.shape
    h = img_shape[0]
    w = img_shape[1]
    out = np.concatenate((im[0:h, 0:w, :, 0],
                          im[0:h, 0:w, :, 1],
                          im[0:h, 0:w, :, 1],
                          im[0:h, 0:w, :, 2]), axis=2)
    return out

def gen_patches_png(img_file):
    # read image
    #img = load_raw_image_packed(file_name)
    img = img_file#imageio.imread(file_name).astype('float')/255.0
    img = np.expand_dims(pack_raw(img), axis=0)
    _,h, w, c = img.shape
    patches = None
    # extract patches
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            x = img[0,i:i + patch_size, j:j + patch_size, :]  # first dim will be removed
            # data aug
            for k in range(0, aug_times):
                # x_aug = data_aug(x, mode=np.random.randint(0, 8))
                if patches is None:
                    patches = x[np.newaxis, :, :, :]  # restore first dim
                else:
                    patches = np.concatenate((patches, x[np.newaxis, :, :, :]), axis=0)  # patches.append(x_aug)
    return patches,[h,w]

def gen_one_patch_png(file_name):
    # read image
    #img = load_raw_image_packed(file_name)
    img = imageio.imread(file_name).astype('float')/255.0
    img = np.expand_dims(pack_raw(img), axis=0)
    _,h, w, c = img.shape
    # extract patches
    i = np.random.randint(0, h - patch_size + 1)
    j = np.random.randint(0, w - patch_size + 1)
    x = img[0,i:i + patch_size, j:j + patch_size, :]  # first dim will be removed
    return np.expand_dims(x,0),[h,w]

def recover_full_img(patches,shape):
    h,w = shape
    out_image = np.zeros((h,w,3))
    index=0
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            out_image[i:i + patch_size,j:j + patch_size,0] = patches[index,:,:,0]
            out_image[i:i + patch_size,j:j + patch_size,1] = patches[index,:,:,1]
            out_image[i:i + patch_size,j:j + patch_size,2] = patches[index,:,:,3]
            index = index+1
    return out_image

def recover_one_img_from_4channel_bayer(patches):
    out_image = patches[0,:,:,(0,1,3)].transpose((1,2,0))
    return out_image

def img_to_array(img):
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')/255.0

    if len(x.shape) == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))
    elif len(x.shape) != 3:
        raise Exception('Unsupported image shape: ', x.shape)
    return x


def load_img(path, target_mode=None, target_size=None):
    from PIL import Image
    img = Image.open(path)
    if target_mode:
        img = img.convert(target_mode)
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img

def pil_image_reader(filepath, target_mode=None, target_size=None, **kwargs):
    img = load_img(filepath, target_mode=target_mode, target_size=target_size)
    return img_to_array(img)

class NumpyArrayGenerator(keras.utils.Sequence):

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None):
        if y is not None and len(X) != len(y):
            raise Exception('X (images tensor) and y (labels) '
                            'should have the same length. '
                            'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        seed = seed or image_data_generator.config['seed']
        self.nb_sample = len(X)
        self.batch_size = batch_size
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
        if (  (index+1)*self.batch_size <= self.nb_sample):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:]
        # Find list of IDs
        # Generate data
        if self.y is None:
            X = self.__data_generation(indexes)
            return X
        else:
            X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nb_sample)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        for i, cur_index in enumerate(list_indexes):
            x = self.X[cur_index]
            x = self.image_data_generator.process(x)
            if i==0:
                batch_x = np.empty((len(list_indexes),)+x.shape)
            batch_x[i] = x
        if self.y is None:
            return batch_x
        batch_y = self.y[list_indexes]
        return batch_x, batch_y


def GAN(filename):
    print('FILENAME: {}'.format(filename))
    img = np.zeros((128,128,3))
    return img

def noiseflow(filename):
    img = np.zeros((128,128,3))
    return img

class DirectoryImageGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, directory, image_data_generator,
                color_mode='rgb', target_size=None,
                image_reader="pil", read_formats=None,
                reader_config=None,
                classes=None, class_mode='categorical',
                batch_size=32, shuffle=True, seed=None):
        self.directory = directory
        self.gan_dir = None
        self.natural_noise_dir = None
        self.image_data_generator = image_data_generator
        self.image_reader = image_reader
        if self.image_reader == 'pil':
            self.image_reader = pil_image_reader
        if read_formats is None:
            read_formats = {'png','jpg','jpeg','bmp'}
        if reader_config is None:
            reader_config = {'target_mode': 'RGB', 'target_size':None}
        self.reader_config = reader_config
        # TODO: move color_mode and target_size to reader_config
        if color_mode == 'rgb':
            self.reader_config['target_mode'] = 'RGB'
            self.n_channels=3
        elif color_mode == 'grayscale':
            self.reader_config['target_mode'] = 'L'
            self.n_channels=1
        elif color_mode == 'rgbfft':
            self.n_channels = 6
        else:
            self.n_channels=1

        self.reader_config['target_size']=None
        if target_size:
            self.target_size = target_size
        else:
            self.target_size = [256,256]

        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode

        seed = seed or image_data_generator.config['seed']

        # first, count the number of samples and classes
        self.nb_sample = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        # if no class is found, add '' for scanning the root folder
        if class_mode is None and len(classes) == 0:
            classes.append('')
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.current_iso = 400
        self.classes_names = ['gaussian','poisson','gaussianpoisson','mosaicgaussianpoisson','noiseflow','GAN','natural']
        self.noise_functions = [generateGaussianRandomVarNoiseChannelSpecific,generatePoissonNoise,generateGaussianPoissonNoise,generateDemosaicWithGaussianPoissonNoise,noiseflow,GAN,None]

        subpath = os.path.join(directory)
        for fname in os.listdir(subpath):
            is_valid = False
            for extension in read_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                self.nb_sample += 1
        print('Found %d images belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        subpath = os.path.join(directory)
        for fname in os.listdir(subpath):
            is_valid = False
            for extension in read_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                self.filenames.append(os.path.join(fname))
                i += 1

        assert len(self.filenames)>0, 'No valid file is found in the target directory.'
        self.reader_config['class_mode'] = self.class_mode
        self.reader_config['classes'] = self.classes
        self.reader_config['filenames'] = self.filenames
        self.reader_config['directory'] = self.directory
        self.reader_config['nb_sample'] = self.nb_sample
        self.reader_config['seed'] = seed
        self.reader_config['sync_seed'] = self.image_data_generator.sync_seed
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        if inspect.isgeneratorfunction(self.image_reader):
            self._reader_generator_mode = True
            self._reader_generator = []
            # set index batch_size to 1
            self.index_generator = self._flow_index(self.N, 1 , self.shuffle, seed)
        else:
            self._reader_generator_mode = False
        self.on_epoch_end()

    def setCurrentISO(self, newISO, noisy_imgs_dir):
        self.natural_noise_dir = noisy_imgs_dir
        
        if newISO not in [400,800,1600,3200]:
            raise 'Pick an ISO level from 400,800,1600,3200'
        else:
            self.current_ISO = newISO

    def setGANdir(self, gan_dir):
        self.gan_dir = gan_dir

    def __len__(self):
        'Denotes the number of batches per epoch'
        lenn = int(np.floor(self.nb_sample) / self.batch_size)
        if (self.nb_sample % self.batch_size >0):
            lenn +=1
            #print('Remainder: {}'.format(self.nb_sample % self.batch_size))
        return lenn


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        self.reader_config['sync_seed'] = self.image_data_generator.sync_seed
        
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



    def __data_generation(self, list_indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_x = np.empty((len(list_indexes), self.target_size[0],self.target_size[1], self.n_channels))
        
        batch_y = np.zeros((len(list_indexes),len(self.classes_names)),dtype='float32')  #multiclass classification
        #batch_y = np.zeros((len(list_indexes),),dtype='float32')   #binary classification
        tmp_NF = np.zeros((256,256,4))
        # Generate data
        for i, cur_index in enumerate(list_indexes):
            fname = self.filenames[cur_index]
            
            #x = srgb_to_linear(x)    
            chosen_index = np.random.randint(0,len(self.noise_functions))
            chosen_noise_function = self.noise_functions[chosen_index]
            if (chosen_noise_function is not None):
                if (chosen_noise_function==GAN):
                    #x = self.image_reader(os.path.join(self.GANdirectory, fname), **self.reader_config)
                    x = self.image_reader(os.path.join(self.gan_dir, fname).replace('.png','_fake_B.png'), **self.reader_config)
                elif (chosen_noise_function==noiseflow):
                    x = self.image_reader(os.path.join(self.directory, fname), **self.reader_config)
                    #patches,img_shape = gen_patches_png(x)
                    #patches = patches**2.2  ##sRGB to linear
                    x = np.stack([x[:,:,0],x[:,:,1],x[:,:,1],x[:,:,2]],axis=-1)  #convert to 4channel bayer
                    x = x**2.2 ##sRGB to linear
                    #cam_iso_idx = random.choice([2,5,8,-1,-3])
                    cam_iso_idx = random.choice(NF_cam_isos[str(self.current_ISO)])
                    row = cam_iso_nlf.iloc[cam_iso_idx]
                    cam = cam_vals.index(row['cam_iso'][:2])
                    iso = float(row['cam_iso'][3:])
                    noise = noise_flow.sample_noise_nf(np.expand_dims(x,0), 0.0, 0.0, self.current_ISO, cam)
                    x_noisy = np.clip(x + noise,0,1)[0]
                    x_noisy = x_noisy**(1/2.2) ##linear to sRGB
                    x_noisy = np.stack([x_noisy[:,:,0],x[:,:,1],x[:,:,-1]],axis=-1) #convert back to 3channel
                    #patches_noisy = patches_noisy**(1/2.2) ##linear to sRGB
                    #recovered_img_noisy = recover_full_img(patches_noisy,img_shape)
                    x = x_noisy
                else:
                    x = self.image_reader(os.path.join(self.directory, fname), **self.reader_config)
                    chosen_std = gaussian_stds[str(self.current_ISO)]['both_dataset']   
                    x = chosen_noise_function(x,chosen_std)
            else:
                x = self.image_reader(os.path.join(self.natural_noise_dir, fname).replace('GT_SRGB','NOISY_SRGB'), **self.reader_config)
            if x.ndim == 2:
                x = np.expand_dims(x, axis=0)
            batch_y[i,chosen_index] = 1.0 

            x = self.image_data_generator.process(x)
            # Store sample
            batch_x[i] = x

            # build batch of labels
        return batch_x, batch_y

