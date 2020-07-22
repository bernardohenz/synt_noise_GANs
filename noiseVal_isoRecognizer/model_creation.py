#########################################################
# Network Architecture
#########################################################
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from keras import optimizers

def create_model(image_size, num_classes=1):
    inputs = Input((image_size[0], image_size[1], 6)) #RGB + RGB_fft

    x = Conv2D(256, kernel_size=(5,5), strides=(1, 1), padding='same')(inputs)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=(5,5), strides=(1, 1), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=(5,5), strides=(1, 1), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Concatenate()([x,x_skip])
    x = MaxPooling2D()(x)

    x_skip = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x_skip)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    #x = Concatenate()([x,x_skip])
    x = MaxPooling2D()(x)

    x_skip = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x_skip)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    #x = Concatenate()([x,x_skip])
    x = MaxPooling2D()(x)

    x_skip = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    #x = Dropout(0.1)(x_skip)
    x = Activation('relu')(x_skip)
    x = MaxPooling2D()(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    #x = Concatenate()([x,x_skip])
    x = Flatten()(x)
    #x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes)(x)
    if num_classes==1:
        x = Activation('sigmoid')(x)
    else:
        x = Activation('softmax')(x)  ##multiclass

    model = Model(inputs, x)
    opt = optimizers.Adam(lr=0.0001)
    loss = 'binary_crossentropy' if num_classes==1 else 'categorical_crossentropy'
    model.compile(loss=loss,    
                  optimizer=opt, metrics=['acc'])
    return model