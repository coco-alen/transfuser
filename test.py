import os
from PIL import Image

import numpy as np
import torch 
import torchvision.transforms as transforms
import torch.nn.functional as F

def scale_and_crop_image(image, scale=1, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    start_x = height//2 - crop//2
    start_y = width//2 - crop//2
    cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image


image_path = "/ssd/dataset/transfuser/2021/data/14_weathers_minimal_data/Town01_long/routes_town01_11_05_20_55_58/rgb_front/0002.png"
scale = 1
input_resolution = 256

image_tensor = torch.from_numpy(np.array(Image.open(image_path)))
print(image_tensor.shape)

image_tensor = torch.from_numpy(np.array(scale_and_crop_image(Image.open(image_path), scale=1, crop=input_resolution)))
print(image_tensor.shape)
image = transforms.ToPILImage()(image_tensor)
image.save('image_into_model1.jpg')

image_tensor = torch.from_numpy(np.array(scale_and_crop_image(Image.open(image_path), scale=0.35, crop=input_resolution)))
print(image_tensor.shape)
image = transforms.ToPILImage()(image_tensor)
image.save('image_into_model2.jpg')