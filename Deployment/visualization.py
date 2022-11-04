#Importing helpers/labels.py from the cityscrapesScripts github page (https://github.com/mcordts/cityscapesScripts)
from labels import *
import numpy as np
from skimage.transform import resize
import cv2

img_height = 256
img_width = 512
batch_size = 16

#Defining visualization functions
def create_mask(pred_mask,img):
        color_map = {
             '0': [0, 0, 0],
             '1': [153, 153, 0],
             '2': [255, 204, 204],
             '3': [255, 0, 127],
             '4': [0, 255, 0],
             '5': [0, 204, 204],
             '6': [255, 0, 0],
             '7': [0, 0, 255]
        }

        dims = (img_height, img_width)
        z = pred_mask
        z = np.squeeze(z)
        z = z.reshape(img_height, img_width, 8)
        z = cv2.resize(z, (dims[1], dims[0]))

        y = np.argmax(z, axis=2)

        img_color = img.copy()   
        for i in range(dims[0]):
            for j in range(dims[1]):
                img_color[i, j] = color_map[str(y[i, j])]
        
        return img_color, y

def convert_to_categ(val):
    return id2label[val].categoryId
    
def convert_categories(mask):
    vector = np.vectorize(convert_to_categ)
    mod_mask = vector(mask)
    return resize(mod_mask, (img_height, img_width))