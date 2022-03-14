import os, sys
from os.path import isfile, join
import numpy as np
from PIL import Image

PATH_INPUT = "input/"
PATH_OUTPUT = "output/"

original_imgs_test = PATH_INPUT

model_path = "model/weights.hdf5"
architechture_path = "model/architecture.json"

import h5py
import numpy as np
from PIL import Image

from os import listdir
from os.path import isfile, join

#Python
import numpy as np
# import ConfigParser

#Keras
from keras.models import model_from_json
from keras.models import Model

import sys
sys.path.insert(0, './lib/')
# help_functions.py
from help_functions import load_hdf5, write_hdf5, rgb2gray, group_images, visualize, masks_Unet, pred_to_imgs

# extract_patches.py
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
from extract_patches import paint_border_overlap
from extract_patches import extract_ordered_overlap
from extract_patches import kill_border
from extract_patches import pred_not_only_FOV
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap

# pre_processing.py
from pre_processing import my_PreProc

def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)

def get_datasets(imgs_dir):
    
    Nimgs = len([f for f in listdir(imgs_dir) if isfile(join(imgs_dir, f))])
    imgs = np.empty((Nimgs,height,width,channels))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            print ("image: " +files[i])
            
            ### write some lines of code to:
            ### - check if image's size is different to requirement, then resize and save to output folder
            ### - else: make a copy to output folder
            
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)    
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    return imgs, files

def get_data(imgs_test, patch_height, patch_width, stride_height, stride_width):
    test_imgs_original = imgs_test
    test_imgs = my_PreProc(test_imgs_original)
    test_imgs = test_imgs[:,:,:,:]
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

#     print "\ntest images shape:"
#     print test_imgs.shape
#     print "range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs))
   
    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)

#     print "\ntest PATCHES images shape:"
#     print patches_imgs_test.shape
#     print "range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test))
    
    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3]
  
def createDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print (directory + " already exists")

channels = 3
height = 2336
width = 1696

# dimension of the patches
patch_height = 32
patch_width = 32

#the stride in case output with average
stride_height = 20
stride_width = 20
assert (stride_height < patch_height and stride_width < patch_width)

#load testing datasets
imgs_test, file_names = get_datasets(original_imgs_test)
print ("test datasets loaded")
print(imgs_test.shape)

#original test images
test_imgs_orig = imgs_test
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]

#Images to patches:
patches_imgs_test = None
new_height = None
new_width = None
masks_test  = None
patches_masks_test = None

patches_imgs_test, new_height, new_width = get_data(
    imgs_test = imgs_test,
    patch_height = patch_height,
    patch_width = patch_width,
    stride_height = stride_height,
    stride_width = stride_width
)

#Load model
model = model_from_json(open(architechture_path).read())
model.load_weights(model_path)

#Calculate the predictions
predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)

#===== Convert the prediction arrays in corresponding images
pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")
pred_imgs = None
pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]

assert(len(file_names) == pred_imgs.shape[0])

N_predicted = pred_imgs.shape[0]

# Save predictions to files
for i in range(int(N_predicted)):
    pred_stripe = group_images(pred_imgs[i:i+1,:,:,:],1)
    file_name =  file_names[i]
    visualize(pred_stripe, "output/" + file_name[0:len(file_name)-4] + "_pred")

print ("All done! Please check the output folder")
