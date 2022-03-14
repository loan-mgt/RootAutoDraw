import os, sys
from os.path import isfile, join
import numpy as np
from PIL import Image

PATH_INPUT = "input/"
PATH_OUTPUT = "output/"

original_imgs_test = PATH_INPUT

model_path = "model/weights.hdf5"
architechture_path = "model/architecture.json"

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
