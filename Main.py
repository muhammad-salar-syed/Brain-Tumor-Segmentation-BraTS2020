
import os
import numpy as np
from Generator import ImageGenerator
from unet_model import UNET3D
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import glob
import random

train_img_dir = "./train/images/"
train_mask_dir = "./train/masks/"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_dir = "./val/images/"
val_mask_dir = "./val/masks/"

val_img_list=os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

num_images = len(os.listdir(train_img_dir))

# Import in batches
batch_size = 1

train_img_datagen = ImageGenerator(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = ImageGenerator(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)


img, mask = train_img_datagen.__next__()

print(img.shape)
print(mask.shape)

#Define loss, metrics and optimizer for training
from keras.optimizers import Adam
from keras.metrics import MeanIoU
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
log_csv = CSVLogger('./brats2020_3dunet_logs.csv', separator=',', append=False)
callbacks_list = [early_stop, log_csv]

metrics = [tf.keras.metrics.MeanIoU(num_classes=4)]
optim = keras.optimizers.Adam()

#Fit the model 
steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size

model = UNET3D(IMG_HEIGHT=128, IMG_WIDTH=128, IMG_DEPTH=128, IMG_CHANNELS=4, num_classes=4)
print(model.input_shape)
print(model.output_shape)

model.compile(optimizer = optim, loss='categorical_crossentropy', metrics=metrics)

history=model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=100,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          callbacks=callbacks_list)

model.save('./brats_3d.hdf5')