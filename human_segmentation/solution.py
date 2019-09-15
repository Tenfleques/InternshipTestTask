import os
try:
	os.chdir(os.path.join(os.getcwd(), 'human_segmentation/notebooks'))
	print(os.getcwd())
except:
	pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from lib import *
import sys
import random
import warnings
# block tensorflow~numpy warnings
warnings.filterwarnings('ignore')

from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
from keras.preprocessing import image


BATCH_SIZE = 10 
# memory issues on 8 gig container
IMG_WIDTH = 240
IMG_HEIGHT = 240
IMG_CHANNELS = 3

TRAIN_PATH = 'data/train'
TRAIN_MASK_PATH = 'data/train_mask'

VALIDATE_PATH = 'data/valid'
VALIDATE_MASK_PATH = 'data/valid_mask'

TEST_PATH = 'data/test'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 83

images = os.listdir(TRAIN_PATH)
val_images = os.listdir(VALIDATE_PATH)
test_images = os.listdir(TEST_PATH)


X_train, Y_train = get_xy(images, TRAIN_PATH, TRAIN_MASK_PATH)
X_validate, Y_validate = get_xy(val_images, VALIDATE_PATH, VALIDATE_MASK_PATH)
X_test, Y_test_placeholder = get_xy(test_images, TEST_PATH)

#%%
random_index = np.random.choice(X_train.shape[0])

#%%
show_img_with_mask(X_train[random_index], Y_train[random_index][:,:,0])

#%%
random_index = np.random.choice(X_validate.shape[0])
show_img_with_mask(X_validate[random_index], Y_validate[random_index][:,:,0])

# Preprocessing
# data augmentation
# shearing of images to overcome overfitting


# Creating the training Image and Mask generator
# image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
# mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')

# # Keep the same seed for image and mask generators so they fit together

# image_datagen.fit(X_train, augment=True, seed=seed)
# mask_datagen.fit(Y_train, augment=True, seed=seed)

# x=image_datagen.flow(X_train,batch_size=BATCH_SIZE,shuffle=True, seed=seed)
# y=mask_datagen.flow(Y_train,batch_size=BATCH_SIZE,shuffle=True, seed=seed)

# # Creating the validation Image and Mask generator
# image_datagen_val = image.ImageDataGenerator()
# mask_datagen_val = image.ImageDataGenerator()

# image_datagen_val.fit(X_validate, augment=True, seed=seed)
# mask_datagen_val.fit(Y_validate, augment=True, seed=seed)

# x_val=image_datagen_val.flow(X_validate,batch_size=BATCH_SIZE,shuffle=True, seed=seed)
# y_val=mask_datagen_val.flow(Y_validate,batch_size=BATCH_SIZE,shuffle=True, seed=seed)

# show_img_with_mask(x.next()[0].astype(np.uint8), np.squeeze(y.next()[0].astype(np.uint8)))

# train_generator = zip(x, y)
# val_generator = zip(x_val, y_val)

# # additional metric 
# # <a href='https://en.wikipedia.org/wiki/Jaccard_index'>Intersection over Union</a>

# def mean_iou(y_true, y_pred):
#     print(y_true)
#     prec = []
#     for t in np.arange(0.5, 1.0, 0.05):
#         y_pred_ = tf.to_int32(y_pred > t)
#         score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
#         K.get_session().run(tf.local_variables_initializer())
#         with tf.control_dependencies([up_opt]):
#             score = tf.identity(score)
#         prec.append(score)
#     return K.mean(K.stack(prec), axis=0)

# # Neural Network ~ U-Net 

# inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
# s = Lambda(lambda x: x / 255) (inputs)

# c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
# c1 = Dropout(0.1) (c1)
# c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
# p1 = MaxPooling2D((2, 2)) (c1)

# c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
# c2 = Dropout(0.1) (c2)
# c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
# p2 = MaxPooling2D((2, 2)) (c2)

# c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
# c3 = Dropout(0.2) (c3)
# c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
# p3 = MaxPooling2D((2, 2)) (c3)

# c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
# c4 = Dropout(0.2) (c4)
# c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
# p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

# c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
# c5 = Dropout(0.3) (c5)
# c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

# u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
# u6 = concatenate([u6, c4])

# c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
# c6 = Dropout(0.2) (c6)
# c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

# u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
# u7 = concatenate([u7, c3])
# c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
# c7 = Dropout(0.2) (c7)
# c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

# u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
# u8 = concatenate([u8, c2])
# c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
# c8 = Dropout(0.1) (c8)
# c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

# u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
# u9 = concatenate([u9, c1], axis=3)
# c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
# c9 = Dropout(0.1) (c9)
# c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

# outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)


# model = Model(inputs=[inputs], outputs=[outputs])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
# model.summary()


# earlystopper = EarlyStopping(patience=3, verbose=1)
# checkpointer = ModelCheckpoint('model-segmentation-1.h5', verbose=1, save_best_only=True)
# results = model.fit_generator(train_generator, validation_data=val_generator, validation_steps=10, steps_per_epoch=250,
#                               epochs=100, callbacks=[earlystopper, checkpointer])



