import numpy as np
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

from PIL import Image

from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout, Reshape, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras import backend as K

import boto3

img_height = 256
img_width = 512
batch_size = 16


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def bilinear_upsample(image_tensor):
    upsampled = tf.image.resize(image_tensor, size=(img_height, img_width))
    return upsampled

def download_weights():
	#Downloading model weights from Amazon S3
	s3 = boto3.resource(service_name='s3',
	                   region_name='eu-west-3',
	                    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
	                    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
	                   )

	bucket = os.environ['S3_BUCKET']

	#Downloading weights in model/weights
	keys=[]
	for obj in s3.Bucket(bucket).objects.all():
	    keys.append(obj.key)

	for key in keys:
	    s3.Bucket(bucket).download_file(Key=key, Filename=key)


def generate_model():
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

	up_stack = [
	    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
	    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
	    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
	    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
	]

	n_classes = 8
	inputs = tf.keras.layers.Input(shape=[img_height, img_width, 3])
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
	model = tf.keras.Model(inputs=inputs, outputs=x)

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[dice_coeff, 'accuracy'])

	download_weights()

	model.load_weights('weights.07.ckpt')

	return model