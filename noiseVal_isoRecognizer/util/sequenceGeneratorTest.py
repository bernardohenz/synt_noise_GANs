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

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras import backend as K
from util.data_aug_transforms import srgb_to_linear

def img_to_array(img, dim_ordering=K.image_dim_ordering()):
    if dim_ordering not in ['th', 'tf']:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')/255.0
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
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

def pil_image_reader(filepath, target_mode=None, target_size=None, dim_ordering=K.image_dim_ordering(), **kwargs):
    img = load_img(filepath, target_mode=target_mode, target_size=target_size)
    return img_to_array(img, dim_ordering=dim_ordering)

class NumpyArrayGenerator(keras.utils.Sequence):

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering=K.image_dim_ordering()):
        if y is not None and len(X) != len(y):
            raise Exception('X (images tensor) and y (labels) '
                            'should have the same length. '
                            'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
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
                dim_ordering=K.image_dim_ordering,
                classes=None, class_mode='categorical',chosen_iso_class='iso0100',
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

        self.dim_ordering = dim_ordering
        self.reader_config['dim_ordering'] = dim_ordering
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
        # if no class is found, add '' for scanning the root folder
        if class_mode is None and len(classes) == 0:
            classes.append('')
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.classes_names = ['iso0100','iso0200','iso0400','iso0800','iso1600','iso3200']
        self.chosen_iso_class = chosen_iso_class
        if (chosen_iso_class in self.classes_names):
            print('Chosen iso class: {}'.format(chosen_iso_class))
        else:
            raise ValueError('Iso class {} is not a valid class, choose one among [iso0100,iso0200,iso0400,iso0800,iso1600,iso3200].'.format(chosen_iso_class))
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
        batch_y = np.zeros((len(list_indexes),6),dtype='float32')   #binary classification
        # Generate data
        for i, cur_index in enumerate(list_indexes):
            fname = self.filenames[cur_index]
            x = self.image_reader(os.path.join(self.directory, fname), **self.reader_config)
            if x.ndim == 2:
                x = np.expand_dims(x, axis=0)
               
            x = self.image_data_generator.process(x)
            # Store sample
            batch_x[i] = x
        chosen_index = self.classes_names.index(self.chosen_iso_class)
        batch_y[:,chosen_index] = 1 #chosen_iso
        # build batch of labels
        return batch_x, batch_y