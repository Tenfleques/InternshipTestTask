"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras

from keras import backend
from keras import utils
from keras.utils import generic_utils

from keras_preprocessing import image



from skimage import data, img_as_float
from skimage import exposure
import scipy.ndimage as ndi

import numpy as np

random_rotation = image.random_rotation
random_shift = image.random_shift
random_shear = image.random_shear
random_zoom = image.random_zoom
apply_channel_shift = image.apply_channel_shift
random_channel_shift = image.random_channel_shift
apply_brightness_shift = image.apply_brightness_shift
random_brightness = image.random_brightness
apply_affine_transform = image.apply_affine_transform
load_img = image.load_img

def transform_matrix_offset_center(matrix, x, y): 
    o_x = float(x) / 2 + 0.5 
    o_y = float(y) / 2 + 0.5 
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]]) 
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix) 
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def array_to_img(x, data_format=None, scale=True, dtype=None):
    if data_format is None:
        data_format = backend.image_data_format()
    if 'dtype' in generic_utils.getargspec(image.array_to_img).args:
        if dtype is None:
            dtype = backend.floatx()
        return image.array_to_img(x,
                                  data_format=data_format,
                                  scale=scale,
                                  dtype=dtype)
    return image.array_to_img(x,
                              data_format=data_format,
                              scale=scale)


def img_to_array(img, data_format=None, dtype=None):
    if data_format is None:
        data_format = backend.image_data_format()
    if 'dtype' in generic_utils.getargspec(image.img_to_array).args:
        if dtype is None:
            dtype = backend.floatx()
        return image.img_to_array(img, data_format=data_format, dtype=dtype)
    return image.img_to_array(img, data_format=data_format)


def save_img(path,
             x,
             data_format=None,
             file_format=None,
             scale=True, **kwargs):
    if data_format is None:
        data_format = backend.image_data_format()
    return image.save_img(path,
                          x,
                          data_format=data_format,
                          file_format=file_format,
                          scale=scale, **kwargs)


class Iterator(image.Iterator, utils.Sequence):
    """Base class for image data iterators.

    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """
    pass


class DirectoryIterator(image.DirectoryIterator, Iterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
            Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
            If set to False, sorts the data in alphanumeric order.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
        dtype: Dtype to use for generated arrays.
    """

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype=None):
        if data_format is None:
            data_format = backend.image_data_format()
        kwargs = {}
        if 'dtype' in generic_utils.getargspec(
                image.ImageDataGenerator.__init__).args:
            if dtype is None:
                dtype = backend.floatx()
            kwargs['dtype'] = dtype
        super(DirectoryIterator, self).__init__(
            directory, image_data_generator,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            **kwargs)


class NumpyArrayIterator(image.NumpyArrayIterator, Iterator):
    """Iterator yielding data from a Numpy array.

    # Arguments
        x: Numpy array of input data or tuple.
            If tuple, the second elements is either
            another numpy array or a list of numpy arrays,
            each of which gets passed
            through as an output without any modifications.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        sample_weight: Numpy array of sample weights.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        dtype: Dtype to use for the generated arrays.
    """

    def __init__(self, x, y, image_data_generator,
                 batch_size=32,
                 shuffle=False,
                 sample_weight=None,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 dtype=None):
        if data_format is None:
            data_format = backend.image_data_format()
        kwargs = {}
        if 'dtype' in generic_utils.getargspec(
                image.NumpyArrayIterator.__init__).args:
            if dtype is None:
                dtype = backend.floatx()
            kwargs['dtype'] = dtype
        super(NumpyArrayIterator, self).__init__(
            x, y, image_data_generator,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_weight=sample_weight,
            seed=seed,
            data_format=data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            **kwargs)


class ImageDataGenerator(image.ImageDataGenerator):
    """Generate batches of tensor image data with real-time data augmentation.
     The data will be looped over (in batches).

    # Arguments
        featurewise_center: Boolean.
            Set input mean to 0 over the dataset, feature-wise.
        samplewise_center: Boolean. Set each sample mean to 0.
        featurewise_std_normalization: Boolean.
            Divide inputs by std of the dataset, feature-wise.
        samplewise_std_normalization: Boolean. Divide each input by its std.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        zca_whitening: Boolean. Apply ZCA whitening.
        rotation_range: Int. Degree range for random rotations.
        width_shift_range: Float, 1-D array-like or int
            - float: fraction of total width, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
                `(-width_shift_range, +width_shift_range)`
            - With `width_shift_range=2` possible values
                are integers `[-1, 0, +1]`,
                same as with `width_shift_range=[-1, 0, +1]`,
                while with `width_shift_range=1.0` possible values are floats
                in the half-open interval `[-1.0, +1.0[`.
        height_shift_range: Float, 1-D array-like or int
            - float: fraction of total height, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
                `(-height_shift_range, +height_shift_range)`
            - With `height_shift_range=2` possible values
                are integers `[-1, 0, +1]`,
                same as with `height_shift_range=[-1, 0, +1]`,
                while with `height_shift_range=1.0` possible values are floats
                in the half-open interval `[-1.0, +1.0[`.
        brightness_range: Tuple or list of two floats. Range for picking
            a brightness shift value from.
        shear_range: Float. Shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range: Float or [lower, upper]. Range for random zoom.
            If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
        channel_shift_range: Float. Range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
            Default is 'nearest'.
            Points outside the boundaries of the input are filled
            according to the given mode:
            - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
            - 'nearest':  aaaaaaaa|abcd|dddddddd
            - 'reflect':  abcddcba|abcd|dcbaabcd
            - 'wrap':  abcdabcd|abcd|abcdabcd
        cval: Float or Int.
            Value used for points outside the boundaries
            when `fill_mode = "constant"`.
        horizontal_flip: Boolean. Randomly flip inputs horizontally.
        vertical_flip: Boolean. Randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None.
            If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (after applying all other transformations).
        preprocessing_function: function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: Image data format,
            either "channels_first" or "channels_last".
            "channels_last" mode means that the images should have shape
            `(samples, height, width, channels)`,
            "channels_first" mode means that the images should have shape
            `(samples, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        validation_split: Float. Fraction of images reserved for validation
            (strictly between 0 and 1).
        dtype: Dtype to use for the generated arrays.

    # Examples
    Example of using `.flow(x, y)`:

    ```python
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)

    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                        steps_per_epoch=len(x_train) / 32, epochs=epochs)

    # here's a more "manual" example
    for e in range(epochs):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
            model.fit(x_batch, y_batch)
            batches += 1
            if batches >= len(x_train) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
    ```
    Example of using `.flow_from_directory(directory)`:

    ```python
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800)
    ```

    Example of transforming images and masks together.

    ```python
    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)

    image_generator = image_datagen.flow_from_directory(
        'data/images',
        class_mode=None,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        'data/masks',
        class_mode=None,
        seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50)
    ```
    """

    def __init__(self,
                 contrast_stretching=False,  # additional
                 histogram_equalization=False, # additional
                 adaptive_equalization=False, # additional
                 featurewise_center=False,                 
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 validation_split=0.0,
                 dtype=None):
        if data_format is None:
            data_format = backend.image_data_format()
        kwargs = {}
        if 'dtype' in generic_utils.getargspec(
                image.ImageDataGenerator.__init__).args:
            if dtype is None:
                dtype = backend.floatx()
            kwargs['dtype'] = dtype
        super(ImageDataGenerator, self).__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            zca_epsilon=zca_epsilon,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format,
            validation_split=validation_split,
            **kwargs)

    def random_transform(self, x):
        """
        """

        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1
        
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0
        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                                [np.sin(theta), np.cos(theta), 0],
                                                [0, 0, 1]])
            transform_matrix = rotation_matrix
        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                            [0, 1, ty],
                                            [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)
        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                            [0, np.cos(shear), 0],
                                            [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                            [0, zy, 0],
                                            [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)
        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                        fill_mode=self.fill_mode, cval=self.cval)
        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)
        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
                        
        if True: # self.contrast_stretching: AttributeError: 'ImageDataGenerator' object has no attribute 'contrast_stretching'
            if np.random.random() < 0.5: #####
                p2, p98 = np.percentile(x, (2, 98)) #####
                x = exposure.rescale_intensity(x, in_range=(p2, p98)) #####
        
        # if self.adaptive_equalization: #####
        #     if np.random.random() < 0.5: #####
        #         x = exposure.equalize_adapthist(x, clip_limit=0.03) #####
                        
        # if self.histogram_equalization: #####
        #     if np.random.random() < 0.5: #####
        #         x = exposure.equalize_hist(x) #####
                        
        return x


array_to_img.__doc__ = image.array_to_img.__doc__
img_to_array.__doc__ = image.img_to_array.__doc__
save_img.__doc__ = image.save_img.__doc__
