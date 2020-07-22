import numpy as np
import time
import math
from keras.layers import Input, Conv2D, Concatenate, Add, Activation, BatchNormalization
from keras.models import Model,Sequential
from tqdm import tqdm

def mse(predictions,targets):
    return np.sum(((predictions - targets) ** 2))/(predictions.shape[0]*predictions.shape[1]*predictions.shape[2])

def cpsnr(img1, img2):
    mse_tmp = mse(np.round(np.clip(img1,0,255)),np.round(np.clip(img2,0,255)))
    PIXEL_MAX = 255.0
    return 10 * math.log10(PIXEL_MAX**2 / mse_tmp)

def predictImg(img,autoencoder):
    img2 = img.astype('float32')[:,:,:3]/255.0
    img2 = np.expand_dims(img2,0)
    start = time.time()
    prediction = autoencoder.predict(img2)
    end = time.time()
    out = np.round(np.clip(prediction[0]*255,0,255))
    return out,end-start

def predictImgNoise(img,autoencoder,noise_std=4):
    img2 = img.astype('float32')[:,:,:3]/255.0
    img2 = np.expand_dims(img2,0)
    img2 = img2+ np.random.normal(0,noise_std/255.0,img2.shape)
    start = time.time()
    prediction = autoencoder.predict(img2)
    end = time.time()
    out = np.round(np.clip(prediction[0]*255,0,255))
    return out,end-start
    
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def predictHugeImg(img,autoencoder,tile_size=1024):
    h,w = img.shape[:2]
    psize=tile_size
    psize = min(min(psize,h),w)
    patch_step = psize
    
    img2 = img.astype('float32')[:,:,:3]/255.0
    
    R = np.zeros(img2.shape, dtype = np.float32)
    rangex = range(0,w,patch_step)
    rangey = range(0,h,patch_step)
    ntiles = len(rangex)*len(rangey)

    for start_x in rangex:
        for start_y in rangey:
            end_x = start_x+psize
            end_y = start_y+psize
            if end_x > w:
                end_x = w
                start_x = end_x-psize
            if end_y > h:
                end_y = h
                start_y = end_y-psize
                
            tileM = img2[start_y:end_y, start_x:end_x, :] 
            tileM = tileM[np.newaxis,:,:,:]
            prediction = autoencoder.predict(tileM)[0]
            s1 = prediction.shape[0]
            s2 = prediction.shape[1]
            R[start_y:start_y+s1,start_x:start_x+s2,:] = prediction
    return np.clip(np.round(R*255.0),0,255)


def create_model(configs):
    ## Model Parameters
    size_NDFA = tuple(configs.size_NDFA)
    size_CFA  = tuple(configs.size_CFA)
    number_of_neutral_density_filters = size_NDFA[0] * size_NDFA[1]
    number_of_colors_on_cfa = size_CFA[0] * size_CFA[1]

    ## Encoder
    main_input= Input(shape=(None, None,3))

    tmp_conv = main_input if configs.train_CFA_only else NeutralDensityLayer(size_NDFA)(main_input)
    submosaics = ColorFilterLayer(size_CFA,kernel_constraint=MaxMax())(tmp_conv)
    if (configs.use_bayer_CFA):
        assert size_CFA == (2,2),"Bayer CFA only works with 2x2 CFA pattern size."
        weights = np.zeros((1,1,3,4))  # (conv_y, conv_x, color_channel, color_filter_index)
        weights[0,0,0,0] = 1 # R
        weights[0,0,1,1] = 1 # G
        weights[0,0,1,2] = 1 # G
        weights[0,0,2,3] = 1 # B
        submosaics = ColorFilterLayer(size_CFA,kernel_constraint=MaxMax(),weights=[weights])(tmp_conv)
    else:
        submosaics = ColorFilterLayer(size_CFA,kernel_constraint=MaxMax())(tmp_conv)

    if not configs.train_CFA_only:
        submosaics = ClippingAndDiscretizing(0.2,4)(submosaics)

    cfa_output_ones=np.ones((1,1,number_of_colors_on_cfa,1))
    mono_mosaic = Conv2D(1,(1,1),padding='same',weights= [cfa_output_ones,np.zeros(1)],trainable=False)(submosaics)

    if (configs.use_interp_layer):
        ## Interpolation Kernel
        interp_kernelA = np.expand_dims(np.append(np.arange(1,size_CFA[0]+1),np.arange(size_CFA[0]-1,0,-1)).astype('float32')/size_CFA[0],0)
        interp_kernelB = np.expand_dims(np.append(np.arange(1,size_CFA[1]+1),np.arange(size_CFA[1]-1,0,-1)).astype('float32')/size_CFA[1],0)
        interp_kernel = np.multiply(np.transpose(interp_kernelA),interp_kernelB)

        all_kernels = np.zeros((number_of_colors_on_cfa,number_of_colors_on_cfa,interp_kernel.shape[0],interp_kernel.shape[1]))
        for i in range(number_of_colors_on_cfa):
            all_kernels[i,i] = interp_kernel


        tmp_interp_output = Conv2D(number_of_colors_on_cfa,(interp_kernel.shape[0],interp_kernel.shape[1]),padding='same',
                                    weights= [all_kernels.transpose((2,3,1,0)),np.zeros(number_of_colors_on_cfa)],
                                    trainable=configs.trainable_interp_layer)(submosaics)

    ## Decoders Input
    decoders_input = mono_mosaic
    if (configs.concatenate_submosaics):
        decoders_input = Concatenate(axis=-1)([decoders_input,submosaics])
    if (configs.use_interp_layer):
        decoders_input = Concatenate(axis=-1)([decoders_input,tmp_interp_output])
    
    decoders_input_model = Model(inputs=main_input,outputs=decoders_input)

    current_layer = Conv2D(configs.number_of_filters, (3, 3),padding='same',activation='relu')(decoders_input)
    last_res_short=current_layer
    for i in range(configs.number_of_residual_blocks):
        current_layer = Conv2D(configs.number_of_filters, (3, 3), padding='same',use_bias=not configs.use_batch_norm)(current_layer)
        if (configs.use_batch_norm):
            current_layer = BatchNormalization(axis=-1)(current_layer)
        current_layer = Activation('relu')(current_layer)
        current_layer = Conv2D(configs.number_of_filters, (3, 3), padding='same',use_bias=not configs.use_batch_norm)(current_layer)
        if (configs.use_batch_norm):
            current_layer = BatchNormalization(axis=-1)(current_layer)
        current_layer = Activation('relu')(current_layer)
        current_layer = Conv2D(configs.number_of_filters, (3, 3), padding='same',use_bias=not configs.use_batch_norm)(current_layer)
        if (configs.use_batch_norm):
            current_layer = BatchNormalization(axis=-1)(current_layer)
        current_layer = Activation('relu')(current_layer)
        if (configs.use_shortcut_on_resblocks):
            current_layer = Add()([current_layer,last_res_short])
        last_res_short = current_layer


    current_layer = Concatenate(axis=-1)([current_layer,decoders_input])
    current_layer = Conv2D(128, (3, 3), padding='same')(current_layer)
    current_layer = Activation('relu')(current_layer)
    current_layer = Conv2D(3, (3, 3), padding='same')(current_layer)

    full_model = Model(inputs=main_input,outputs=current_layer)
    return full_model