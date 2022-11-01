from flask import Flask, render_template, request
#Importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Setting large figure size for Seaborn
sns.set(rc={'figure.figsize':(11.7,8.27),"font.size":20,"axes.titlesize":20,"axes.labelsize":18})

import dill
import os
import shutil

from skimage import io
import cv2

from skimage.transform import resize

import tensorflow as tf

tf.get_logger().setLevel('ERROR')
import tensorflow_hub as hub

from PIL import Image

from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout, Reshape, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras import backend as K

#Importing helpers/labels.py from the cityscrapesScripts github page (https://github.com/mcordts/cityscapesScripts)
from labels import *

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


cats = {'void': [0, 1, 2, 3, 4, 5, 6],
 'flat': [7, 8, 9, 10],
 'construction': [11, 12, 13, 14, 15, 16],
 'object': [17, 18, 19, 20],
 'nature': [21, 22],
 'sky': [23],
 'human': [24, 25],
 'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]}



def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def total_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + (3*dice_loss(y_true, y_pred))
    return loss

img_height = 256
img_width = 256
batch_size = 16

base_model = tf.keras.applications.MobileNetV2(input_shape=[img_height, img_width, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

from tensorflow_examples.models.pix2pix import pix2pix

from IPython.display import clear_output
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

n_classes = 8
def bilinear_upsample(image_tensor):
    upsampled = tf.image.resize(image_tensor, size=(img_height, img_width))
    return upsampled

def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[img_height, img_width, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of unet
    last = tf.keras.layers.Conv2DTranspose(
      filters=n_classes, kernel_size=3,
      padding='same')  #64x64 -> 128x128

    x = last(x)
    
    #Adding image segmentation layers
    x = Lambda(bilinear_upsample, name='bilinear_upsample')(x)
    x = Reshape((img_height*img_width, n_classes))(x)
    x = Activation('softmax', name='final_softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

OUTPUT_CLASSES = 8

model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[dice_coeff, 'accuracy'])

model.load_weights('weights.28.ckpt')

target_img = os.path.join(os.getcwd() , 'static/images')

root_dir = "/home/faskill/Files/0. Files/1. WIP/2. Data Analysis/Openclassrooms/AI Engineer/Project 8/"
img_dir = "Data/photos_raw/test/"
mask_dir = "Data/masks/test/"


with open('list/test_image_list.pkl', 'rb') as file:
    test_image_list = dill.load(file)
with open('list/test_mask_list.pkl', 'rb') as file:
    test_mask_list = dill.load(file)


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

@app.route('/')
def index_view():
    return render_template('index.html', list = test_image_list)

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):

    img = load_img(filename, target_size=(256, 256))
    x = tf.keras.utils.img_to_array(img)
    y = np.expand_dims(x, axis=0)
    return y, x

@app.route("/predict" , methods=['GET', 'POST'])
def predict():
    
    image_name = request.form.get('file')
    file_path = root_dir + img_dir + image_name
    mask_path = root_dir + mask_dir + image_name

    img = resize(io.imread(file_path), (img_height, img_width))
    true_mask = convert_categories(io.imread(mask_path))
    rgb_mask, grayscale_mask = create_mask(model.predict(img[tf.newaxis,...]), img)

    image_path = os.path.join('static/images', 'photo.png')
    pred_mask_path = os.path.join('static/images', 'predicted_mask.png')
    true_mask_path = os.path.join('static/images', 'true_mask.png')

    plt.imsave(pred_mask_path, grayscale_mask.astype('uint8'))
    plt.imsave(image_path, img)
    plt.imsave(true_mask_path, true_mask)

    return render_template('predict.html', user_image = image_path, true_mask=true_mask_path,
        pred_mask=pred_mask_path, list = test_image_list)

@app.route('/custom',methods=['GET','POST'])
def custom():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', 'photo.png')
            file.save(file_path)
            img = resize(io.imread(file_path), (img_height, img_width))

            rgb_mask, grayscale_mask = create_mask(model.predict(img[tf.newaxis,...]), img)

            pred_mask_path = os.path.join('static/images', 'predicted_mask.png')
            plt.imsave(pred_mask_path, grayscale_mask.astype('uint8'))
            plt.imsave(file_path, img)
            

            return render_template('predict.html', user_image = file_path, 
                pred_mask=pred_mask_path, true_mask="none", list = test_image_list)
                
        else:
            return "Unable to read the file. Please check file extension"




if __name__ == '__main__':
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.run(debug=True)
