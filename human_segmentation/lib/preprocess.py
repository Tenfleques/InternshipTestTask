from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.preprocessing import image


def shear_augmentation_image():
    """
    """
    x=None
    y=None

    # Creating the training Image and Mask generator
    image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
    mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')

    seed = 83
    image_datagen.fit(x_, augment=True, seed=seed)
    x=image_datagen.flow(x_,batch_size=BATCH_SIZE,shuffle=True, seed=seed)

    if(y_):
        mask_datagen.fit(y_, augment=True, seed=seed)
        y=mask_datagen.flow(y_,batch_size=BATCH_SIZE,shuffle=True, seed=seed)

    return (x,y)