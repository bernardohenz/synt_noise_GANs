import keras
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract
from keras.models import Model, load_model
import glob, os, time, random, math, sys
import scipy
import keras.backend as K
import numpy as np
from tqdm import tqdm
from utils.utils_test import cpsnr, predictHugeImg
from PIL import Image


def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true))/2

def DnCNN(depth,filters=64,image_channels=3, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(None,None,image_channels),name = 'input'+str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',name = 'conv'+str(layer_count))(inpt)
    layer_count += 1
    x = Activation('relu',name = 'relu'+str(layer_count))(x)
    # depth-2 layers, Conv+BN+relu
    for i in range(depth-2):
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            #x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x) 
            x = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
        layer_count += 1
        x = Activation('relu',name = 'relu'+str(layer_count))(x)  
    # last layer, Conv
    layer_count += 1
    x = Conv2D(filters=image_channels, kernel_size=(3,3), strides=(1,1), kernel_initializer='Orthogonal',padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
    layer_count += 1
    x = Subtract(name = 'subtract' + str(layer_count))([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    
    return model

model = DnCNN(depth=17, filters=64, image_channels=3, use_bnorm=True)
model_names = glob.glob('trained_models/*.h5')

total_number_of_patches = 40+40+40+74+84

for cur_model_name in model_names:
   model.load_weights(cur_model_name)
   with tqdm(total=total_number_of_patches) as pbar:
      ##RENOIR dataset
      renoir_data_dir = '/media/bernardo/Storage/RENOIR_Dataset/t3i'
      img_names = glob.glob(os.path.join(renoir_data_dir,'Ba*'))

      renoir_t3i_psnrs = np.zeros((len(img_names)))
      for i,img_name in enumerate(img_names):
         imgs_noisy_names = glob.glob(os.path.join(img_name,'*Noisy.bmp'))
         img_noisy = np.asarray(Image.open( imgs_noisy_names[0] )).astype('float32')
         imgs_clean_names = glob.glob(os.path.join(img_name,'*Reference.bmp'))
         img_clean = np.asarray(Image.open( imgs_clean_names[0] )).astype('float32')
         img_predicted = predictHugeImg(img_noisy,model)
         renoir_t3i_psnrs[i] = cpsnr(img_clean,img_predicted)
         pbar.update(1)

      renoir_data_dir = '/media/bernardo/Storage/RENOIR_Dataset/Mi3_Aligned'
      img_names = glob.glob(os.path.join(renoir_data_dir,'Ba*'))

      renoir_mi3_psnrs = np.zeros((len(img_names)))
      for i,img_name in enumerate(img_names):
         imgs_noisy_names = glob.glob(os.path.join(img_name,'*Noisy.bmp'))
         img_noisy = np.asarray(Image.open( imgs_noisy_names[0] )).astype('float32')
         imgs_clean_names = glob.glob(os.path.join(img_name,'*Reference.bmp'))
         img_clean = np.asarray(Image.open( imgs_clean_names[0] )).astype('float32')
         img_predicted = predictHugeImg(img_noisy,model)
         renoir_mi3_psnrs[i] = cpsnr(img_clean,img_predicted)
         pbar.update(1)

      renoir_data_dir = '/media/bernardo/Storage/RENOIR_Dataset/S90_Aligned'
      img_names = glob.glob(os.path.join(renoir_data_dir,'Ba*'))

      renoir_s90_psnrs = np.zeros((len(img_names)))
      for i,img_name in enumerate(img_names):
         imgs_noisy_names = glob.glob(os.path.join(img_name,'*Noisy.bmp'))
         img_noisy = np.asarray(Image.open( imgs_noisy_names[0] )).astype('float32')
         imgs_clean_names = glob.glob(os.path.join(img_name,'*Reference.bmp'))
         img_clean = np.asarray(Image.open( imgs_clean_names[0] )).astype('float32')
         img_predicted = predictHugeImg(img_noisy,model)
         renoir_s90_psnrs[i] = cpsnr(img_clean,img_predicted)
         pbar.update(1)

      ## SIDD dataset
      sidd_data_dir = '/media/bernardo/Storage/SIDD_Small_sRGB_Only'
      img_namesL = glob.glob(os.path.join(sidd_data_dir,'Data/*_L'))
      img_namesN = glob.glob(os.path.join(sidd_data_dir,'Data/*_N'))

      psnrsL = np.zeros((len(img_namesL)))
      for i,img_name in enumerate(img_namesL):
         imgs_noisy_name = glob.glob(os.path.join(img_name,'NOISY*'))
         img_noisy = np.asarray(Image.open( imgs_noisy_name[0] )).astype('float32')
         imgs_clean_names = glob.glob(os.path.join(img_name,'GT*'))
         img_clean = np.asarray(Image.open( imgs_clean_names[0] )).astype('float32')
         img_predicted = predictHugeImg(img_noisy,model)
         psnrsL[i] = cpsnr(img_clean,img_predicted)
         pbar.update(1)

      psnrsN = np.zeros((len(img_namesN)))
      for i,img_name in enumerate(img_namesN):
         imgs_noisy_name = glob.glob(os.path.join(img_name,'NOISY*'))
         img_noisy = np.asarray(Image.open( imgs_noisy_name[0] )).astype('float32')
         imgs_clean_names = glob.glob(os.path.join(img_name,'GT*'))
         img_clean = np.asarray(Image.open( imgs_clean_names[0] )).astype('float32')
         img_predicted = predictHugeImg(img_noisy,model)
         psnrsN[i] = cpsnr(img_clean,img_predicted)
         pbar.update(1)

   print("Model: {}".format(cur_model_name))
   print('Renoir_t3i: {:.2f}'.format(renoir_t3i_psnrs.mean()))
   print('Renoir_mi3: {:.2f}'.format(renoir_mi3_psnrs.mean()))
   print('Renoir_s90: {:.2f}'.format(renoir_s90_psnrs.mean()))
   print('SSID_L: {:.2f}'.format(psnrsL.mean()))
   print('SSID_N: {:.2f}'.format(psnrsN.mean()))
   print("======================================")
