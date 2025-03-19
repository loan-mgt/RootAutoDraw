import os, sys
from os import listdir
from os.path import isfile, join

from PIL import Image
import h5py
import numpy as np

# Keras
import tensorflow as tf

sys.path.insert(0, './lib/')
from help_functions import load_hdf5, write_hdf5, rgb2gray, group_images, visualize, masks_Unet, pred_to_imgs
from extract_patches import recompone, recompone_overlap, paint_border, paint_border_overlap, extract_ordered_overlap, \
    kill_border, pred_not_only_FOV, get_data_testing, get_data_testing_overlap
from pre_processing import my_PreProc

PATH_INPUT = "input/"
PATH_OUTPUT = "output/"

original_imgs_test = PATH_INPUT

model_path = "model/weights.keras"
architechture_path = "model/architecture.json"

os.makedirs("output", exist_ok=True)


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def get_datasets(imgs_dir):
    Nimgs = len([f for f in listdir(imgs_dir) if isfile(join(imgs_dir, f))])
    imgs = np.empty((Nimgs, height, width, channels))
    for path, subdirs, files in os.walk(imgs_dir):  # list all files, directories in the path
        for i in range(len(files)):
            print("image: " + files[i])
            img = Image.open(imgs_dir + files[i])
            imgs[i] = np.asarray(img)
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    assert (imgs.shape == (Nimgs, channels, height, width))
    return imgs, files


def get_image(image_pil):
    image_as_array = np.asarray(image_pil)
    image_as_array = np.expand_dims(image_as_array, 0)
    image_as_array = np.transpose(image_as_array, (0, 3, 1, 2))
    print(image_as_array.shape)
    return image_as_array


def get_data(imgs_test, patch_height, patch_width, stride_height, stride_width):
    test_imgs_original = imgs_test
    test_imgs = my_PreProc(test_imgs_original)
    test_imgs = test_imgs[:, :, :, :]
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    # extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3]


def createDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print(directory + " already exists")


if __name__ == '__main__':
    # Load model  
    model = tf.keras.models.load_model(model_path)
    
    for filename in listdir("input"):
        if filename == ".ipynb_checkpoints":
            continue
        path_image = join("input", filename)
        image = Image.open(path_image)
        width, height = image.size
        channels = 3

        # dimension of the patches
        patch_size = 32
        stride = 20

        image_array = get_image(image)

        patches_imgs_test, new_height, new_width = get_data(
            imgs_test=image_array,
            patch_height=patch_size,
            patch_width=patch_size,
            stride_height=stride,
            stride_width=stride
        )

        # Calculate the predictions
        predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)

        # ===== Convert the prediction arrays in corresponding images
        pred_patches = pred_to_imgs(predictions, patch_size, patch_size, "original")
        pred_imgs = None
        pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride, stride)  # predictions
        pred_imgs = pred_imgs[:, :, 0:height, 0:width]
        pred_stripe = group_images(pred_imgs[:, :, :, :], 1)
        visualize(pred_stripe, "output/" + filename[0:-4] + "_pred")
        
    print("All done! Please check the output folder")
