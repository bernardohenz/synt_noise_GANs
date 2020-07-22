'''Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new process methods, etc...

Based on the implementation of git user oeway (Wei Ouyang). Link: https://github.com/oeway/keras
'''
from __future__ import absolute_import
from __future__ import print_function

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

from keras import backend as K
from util.sequenceGenerator import DirectoryImageGenerator,NumpyArrayGenerator
from keras.utils.generic_utils import Progbar
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def random_rotation(x, rg, row_index=1, col_index=2, channel_index=0,
                    fill_mode='nearest', cval=0.):
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_shift(x, wrg, hrg, row_index=1, col_index=2, channel_index=0,
                 fill_mode='nearest', cval=0.):
    h, w = x.shape[row_index], x.shape[col_index]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_shear(x, intensity, row_index=1, col_index=2, channel_index=0,
                 fill_mode='nearest', cval=0.):
    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_zoom(x, zoom_range, row_index=1, col_index=2, channel_index=0,
                fill_mode='nearest', cval=0.):
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_barrel_transform(x, intensity):
    # TODO
    pass


def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def array_to_img(x, mode=None, scale=True):
    from PIL import Image
    x = x.copy()
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3 and mode == 'RGB':
        return Image.fromarray(x.astype('uint8'), mode)
    elif x.shape[2] == 1 and mode == 'L':
        return Image.fromarray(x[:, :, 0].astype('uint8'), mode)
    elif mode:
        return Image.fromarray(x, mode)
    else:
        raise Exception('Unsupported array shape: ', x.shape)


def img_to_array(img):
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')/255.0
    if len(x.shape) == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))
    elif len(x.shape)!=3:
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

def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]

def pil_image_reader(filepath, target_mode=None, target_size=None, **kwargs):
    img = load_img(filepath, target_mode=target_mode, target_size=target_size)
    return img_to_array(img)

def standardize(x,
                rescale=False,
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                mean=None, std=None,
                samplewise_std_normalization=False,
                zca_whitening=False, principal_components=None,
                featurewise_standardize_axis=None,
                samplewise_standardize_axis=None,
                fitting=False,
                verbose=0,
                config={},
                **kwargs):
    '''

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        featurewise_standardize_axis: axis along which to perform feature-wise center and std normalization.
        samplewise_standardize_axis: axis along which to to perform sample-wise center and std normalization.
        zca_whitening: apply ZCA whitening.

    '''
    if fitting:
        if config.has_key('_X'):
            # add data to _X array
            config['_X'][config['_iX']] = x
            config['_iX'] +=1
            if verbose and config.has_key('_fit_progressbar'):
                config['_fit_progressbar'].update(config['_iX'], force=(config['_iX']==fitting))

            # the array (_X) is ready to fit
            if config['_iX'] >= fitting:
                X = config['_X'].astype('float32')
                del config['_X']
                del config['_iX']
                if featurewise_center or featurewise_std_normalization:
                    featurewise_standardize_axis = featurewise_standardize_axis or 0
                    if type(featurewise_standardize_axis) is int:
                        featurewise_standardize_axis = (featurewise_standardize_axis, )
                    assert 0 in featurewise_standardize_axis, 'feature-wise standardize axis should include 0'

                if featurewise_center:
                    mean = np.mean(X, axis=featurewise_standardize_axis, keepdims=True)
                    config['mean'] = np.squeeze(mean, axis=0)
                    X -= mean

                if featurewise_std_normalization:
                    std = np.std(X, axis=featurewise_standardize_axis, keepdims=True)
                    config['std'] = np.squeeze(std, axis=0)
                    X /= (std + 1e-7)

                if zca_whitening:
                    flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
                    sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
                    U, S, V = linalg.svd(sigma)
                    config['principal_components'] = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)
                if verbose:
                    del config['_fit_progressbar']
        else:
            # start a new fitting, fitting = total sample number
            config['_X'] = np.zeros((fitting,)+x.shape)
            config['_iX'] = 0
            config['_X'][config['_iX']] = x
            config['_iX'] +=1
            if verbose:
                config['_fit_progressbar'] = Progbar(target=fitting, verbose=verbose)
        return x

    if rescale:
        x *= rescale

    # x is a single image, so it doesn't have image number at index 0
    channel_index = 2

    samplewise_standardize_axis = samplewise_standardize_axis or channel_index
    if type(samplewise_standardize_axis) is int:
        samplewise_standardize_axis = (samplewise_standardize_axis, )

    if samplewise_center:
        x -= np.mean(x, axis=samplewise_standardize_axis, keepdims=True)
    if samplewise_std_normalization:
        x /= (np.std(x, axis=samplewise_standardize_axis, keepdims=True) + 1e-7)

    if verbose:
        if (featurewise_center and mean is None) or (featurewise_std_normalization and std is None) or (zca_whitening and principal_components is None):
            print('WARNING: feature-wise standardization and zca whitening will be disabled, please run "fit" first.')

    if featurewise_center:
        if mean is not None:
            x -= mean
    if featurewise_std_normalization:
        if std is not None:
            x /= (std + 1e-7)

    if zca_whitening:
        if principal_components is not None:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
    return x


def random_transform(x,
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     shear_range=0.,
                     zoom_range=0.,
                     channel_shift_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     horizontal_flip=False,
                     vertical_flip=False,
                     rescale=None,
                     sync_seed=None,
                     **kwargs):
    '''

    # Arguments
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation).
    '''
    np.random.seed(sync_seed)

    x = x.astype('float32')
    # x is a single image, so it doesn't have image number at index 0

    img_channel_index = 2
    img_row_index = 0
    img_col_index = 1
    # use composition of homographies to generate final transform that needs to be applied
    if rotation_range:
        theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
    else:
        theta = 0
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    if height_shift_range:
        tx = np.random.uniform(-height_shift_range, height_shift_range) * x.shape[img_row_index]
    else:
        tx = 0

    if width_shift_range:
        ty = np.random.uniform(-width_shift_range, width_shift_range) * x.shape[img_col_index]
    else:
        ty = 0

    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    if shear_range:
        shear = np.random.uniform(-shear_range, shear_range)
    else:
        shear = 0
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    if np.isscalar(zoom_range):
        zoom_range = [1 - zoom_range, 1 + zoom_range]
    elif len(zoom_range) == 2:
        zoom_range = [zoom_range[0], zoom_range[1]]
    else:
        raise Exception('zoom_range should be a float or '
                        'a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

    h, w = x.shape[img_row_index], x.shape[img_col_index]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    x = apply_transform(x, transform_matrix, img_channel_index,
                        fill_mode=fill_mode, cval=cval)
    if channel_shift_range != 0:
        x = random_channel_shift(x, channel_shift_range, img_channel_index)

    if horizontal_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_col_index)

    if vertical_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_row_index)

    # TODO:
    # barrel/fisheye

    np.random.seed()
    return x

class ImageDataGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        featurewise_standardize_axis: axis along which to perform feature-wise center and std normalization.
        samplewise_standardize_axis: axis along which to to perform sample-wise center and std normalization.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation).
        seed: random seed for reproducible pipeline processing. If not None, it will also be used by `flow` or
            `flow_from_directory` to generate the shuffle index in case of no seed is set.
    '''
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 featurewise_standardize_axis=None,
                 samplewise_standardize_axis=None,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 seed=None,
                 verbose=1):
        self.config = copy.deepcopy(locals())
        self.config['config'] = self.config
        self.config['mean'] = None
        self.config['std'] = None
        self.config['principal_components'] = None
        self.config['rescale'] = rescale


        self.__sync_seed = self.config['seed'] or np.random.randint(0, 4294967295,dtype=np.int64)

        self.default_pipeline = []
        self.default_pipeline.append(random_transform)
        self.default_pipeline.append(standardize)
        self.set_pipeline(self.default_pipeline)

        self.__fitting = False
        self.fit_lock = threading.Lock()

    @property
    def sync_seed(self):
        return self.__sync_seed

    @property
    def fitting(self):
        return self.__fitting

    @property
    def pipeline(self):
        return self.__pipeline

    def sync(self, image_data_generator):
        self.__sync_seed = image_data_generator.sync_seed
        return (self, image_data_generator)

    def set_pipeline(self, p):
        if p is None:
            self.__pipeline = self.default_pipeline
        elif type(p) is list:
            self.__pipeline = p
        else:
            raise Exception('invalid pipeline.')

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None):
        return NumpyArrayGenerator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed)

    def flow_from_directory(self, directory,
                            color_mode='rgb', target_size=(128,128),
                            image_reader='pil', reader_config=None,
                            read_formats=None,
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None):
        if reader_config is None:
            reader_config={'target_mode':'RGB'}
        if read_formats is None:
            read_formats={'png','jpg','jpeg','bmp'}
        return DirectoryImageGenerator(directory, self,
                                classes=classes, class_mode=class_mode,
                                color_mode=color_mode, target_size=target_size,
                                image_reader=image_reader, reader_config=reader_config,
                                read_formats=read_formats,
                                batch_size=batch_size, shuffle=shuffle, seed=seed)

    def process(self, x):
        # get next sync_seed
        np.random.seed(self.__sync_seed)
        self.__sync_seed = np.random.randint(0, 4294967295,dtype=np.int64)
        self.config['fitting'] = self.__fitting
        self.config['sync_seed'] = self.__sync_seed
        for p in self.__pipeline:
            x = p(x, **self.config)
        return x

    def fit_generator(self, generator, nb_iter):
        '''Fit a generator

        # Arguments
            generator: Iterator, generate data for fitting.
            nb_iter: Int, number of iteration to fit.
        '''
        with self.fit_lock:
            try:
                self.__fitting = nb_iter*generator.batch_size
                for i in range(nb_iter):
                    next(generator)
            finally:
                self.__fitting = False

    def fit(self, X, rounds=1):
        '''Fit the pipeline on a numpy array

        # Arguments
            X: Numpy array, the data to fit on.
            rounds: how many rounds of fit to do over the data
        '''
        X = np.copy(X)
        with self.fit_lock:
            try:
                self.__fitting = rounds*X.shape[0]
                for r in range(rounds):
                    for i in range(X.shape[0]):
                        self.process(X[i])
            finally:
                self.__fitting = False





