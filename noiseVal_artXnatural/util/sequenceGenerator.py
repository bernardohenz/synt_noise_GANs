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



class DirectoryImageGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, directory, image_data_generator,
                color_mode='rgb', target_size=None,
                image_reader="pil", read_formats=None,
                reader_config=None,
                classes=None, class_mode='categorical',
                batch_size=32, shuffle=True, seed=None):
        self.directory = directory
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
            self.target_size = [128,128]

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
        self.classes_names = ['artificial','natural'] #['gaussian','poisson','gaussianpoisson','mosaicgaussianpoisson','natural']
        self.natural_noise_dirs = None
        self.noise_functions = [generateGaussianRandomVarNoiseChannelSpecific,generatePoissonNoise,generateGaussianPoissonNoise,generateDemosaicWithGaussianPoissonNoise, None]
        self.current_ISO = 400

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
        self.natural_noise_dirs = noisy_imgs_dir
        
        if newISO not in [400,800,1600,3200]:
            raise 'Pick an ISO level from 400,800,1600,3200'
        else:
            self.current_ISO = newISO



    def __len__(self):
        'Denotes the number of batches per epoch'
        lenn = int(np.floor(self.nb_sample) / self.batch_size)
        if (self.nb_sample % self.batch_size >0):
            lenn +=1
            #print('Remainder: {}'.format(self.nb_sample % self.batch_size))
        return lenn

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

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_x = np.empty((len(list_indexes), self.target_size[0],self.target_size[1], self.n_channels))
        
        #batch_y = np.zeros((len(index_array),len(self.noise_functions)),dtype='float32')  #multiclass classification
        batch_y = np.zeros((len(list_indexes),),dtype='float32')   #binary classification
        # Generate data
        for i, cur_index in enumerate(list_indexes):
            fname = self.filenames[cur_index]
            x = self.image_reader(os.path.join(self.directory, fname), **self.reader_config)
            if x.ndim == 2:
                x = np.expand_dims(x, axis=0)
            
            #x = srgb_to_linear(x)    
            choosen_index = np.random.randint(0,len(self.noise_functions))
            choosen_noise_function = self.noise_functions[choosen_index]
            if (choosen_noise_function is not None):
                #chosen_iso = random.choice(list(gaussian_stds.keys()))
                #chosen_dataset = random.choice(list(gaussian_stds['400'].keys()))
                chosen_std = gaussian_stds[str(self.current_ISO)]['both_dataset']
                x = choosen_noise_function(x,chosen_std)
            else:    
                x = self.image_reader(os.path.join(random.choice(self.natural_noise_dirs), fname.replace('GT_SRGB','NOISY_SRGB')), **self.reader_config)
                batch_y[i] = 1.0    #binary classification
            #batch_y[i,choosen_index] = 1.0  #multiclass
            #batch_y[i]= choosen_index 
            x = self.image_data_generator.process(x)
            # Store sample
            batch_x[i] = x

            # build batch of labels
        return batch_x, batch_y

        if self.class_mode == 'sparse':
            batch_y = self.classes[list_indexes]
        elif self.class_mode == 'binary':
            batch_y = self.classes[list_indexes].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[list_indexes]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y