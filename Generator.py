
import os
import numpy as np
from keras.utils import to_categorical
from patchify import patchify, unpatchify

def image_loader(directory,image_list):
    images=[]
    for i, name in enumerate(image_list):    
        if (name.split('.')[1] == 'npy'):
            img = np.load(directory+name)
            images.append(img)
    images = np.array(images)
    return(images)




def ImageGenerator(directory, image_list, mask_directory, mask_list, batch_size):

    while True:
        start = 0
        end = batch_size

        while start < len(image_list):
            limit = min(end, len(image_list))
                       
            X = image_loader(directory, image_list[start:limit])
            Y = image_loader(mask_directory, mask_list[start:limit])

            yield (X,Y)   

            start += batch_size   
            end += batch_size
            
