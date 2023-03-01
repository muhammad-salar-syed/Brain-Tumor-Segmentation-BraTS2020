
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, Concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.metrics import MeanIoU


def UNET3D(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS,num_classes):

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH,IMG_CHANNELS))

    #Contraction path
    CB1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(inputs)
    CB1 = Dropout(0.1)(CB1)
    CB1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(CB1)
    P1 = MaxPooling3D((2, 2, 2))(CB1)
    
    CB2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(P1)
    CB2 = Dropout(0.1)(CB2)
    CB2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(CB2)
    P2 = MaxPooling3D((2, 2, 2))(CB2)
     
    CB3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(P2)
    CB3 = Dropout(0.2)(CB3)
    CB3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(CB3)
    P3 = MaxPooling3D((2, 2, 2))(CB3)
     
    CB4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(P3)
    CB4 = Dropout(0.2)(CB4)
    CB4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(CB4)
    P4 = MaxPooling3D(pool_size=(2, 2, 2))(CB4)
     
    CB5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(P4)
    CB5 = Dropout(0.3)(CB5)
    CB5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(CB5)
    
    #Expansive path 
    EB4 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(CB5)
    EB4 = concatenate([EB4, CB4])
    EB4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(EB4)
    EB4 = Dropout(0.2)(EB4)
    EB4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(EB4)
     
    EB3 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(EB4)
    EB3 = concatenate([EB3, CB3])
    EB3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(EB3)
    EB3 = Dropout(0.2)(EB3)
    EB3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(EB3)
     
    EB2 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(EB3)
    EB2 = concatenate([EB2, CB2])
    EB2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(EB2)
    EB2 = Dropout(0.1)(EB2)
    EB2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(EB2)
     
    EB1 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(EB2)
    EB1 = concatenate([EB1, CB1])
    EB1 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(EB1)
    EB1 = Dropout(0.1)(EB1)
    EB1 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(EB1)
     
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(EB1)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    
    return model
