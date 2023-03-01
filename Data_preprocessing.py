
import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Training dataset

t1_section = sorted(glob.glob('./BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1.nii'))
t2_section = sorted(glob.glob('./BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
t1ce_section = sorted(glob.glob('./BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
flair_section = sorted(glob.glob('./BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
mask = sorted(glob.glob('./BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*seg.nii'))

for i in range(len(t1_section)):  
      
    t1_image=nib.load(t1_section[i]).get_fdata()
    t1_image=scaler.fit_transform(t1_image.reshape(-1, t1_image.shape[-1])).reshape(t1_image.shape)
   
    t1ce_image=nib.load(t1ce_section[i]).get_fdata()
    t1ce_image=scaler.fit_transform(t1ce_image.reshape(-1, t1ce_image.shape[-1])).reshape(t1ce_image.shape)
   
    flair_image=nib.load(flair_section[i]).get_fdata()
    flair_image=scaler.fit_transform(flair_image.reshape(-1, flair_image.shape[-1])).reshape(flair_image.shape)
        
    t2_image=nib.load(t2_section[i]).get_fdata()
    t2_image=scaler.fit_transform(t2_image.reshape(-1, t2_image.shape[-1])).reshape(t2_image.shape)
    
    Mask=nib.load(mask[i]).get_fdata()
    Mask=Mask.astype(np.uint8)
    Mask[Mask==4] = 3  
    
    combined_images = np.stack([t1_image,t1ce_image,flair_image,t2_image], axis=3)
    
    #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
    combined_images=combined_images[56:184, 56:184, 13:141]
    Mask = Mask[56:184, 56:184, 13:141]
    
    val, counts = np.unique(Mask, return_counts=True)
    
    if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
        print("Save Me")
        Mask= to_categorical(Mask, num_classes=4)
        np.save('./images/image_'+str(i)+'.npy', combined_images)
        np.save('./masks/mask_'+str(i)+'.npy', Mask)
        
    else:
        print("I am useless")   

##########################################################################
import splitfolders 

input_folder = './Data/'
output_folder = './Data/split/'

# Split with a ratio.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.8, .2), group_prefix=None)