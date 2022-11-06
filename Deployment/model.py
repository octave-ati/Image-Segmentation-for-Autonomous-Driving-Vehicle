import os
import shutil
import segmentation_models as sm
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout, Reshape, Lambda
from tensorflow.keras import Model

import boto3

img_height = 256
img_width = 512
batch_size = 16


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

	img_height = 256
	img_width = 512
	n_classes = 8

	ln = sm.Linknet(backbone_name='resnet18', encoder_weights='imagenet', input_shape= (256,512,3), classes=8)

	x = ln.layers[-2].output
	x = Reshape((img_height*img_width, n_classes))(x)
	x = Activation('softmax', name='final_softmax')(x)

	linknet = Model(inputs=ln.input, outputs=x)

	init_lr = 3e-5

	optimizer = tf.keras.optimizers.experimental.AdamW()

	linknet.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	download_weights()

	linknet.load_weights('weights.29.ckpt')

	return linknet