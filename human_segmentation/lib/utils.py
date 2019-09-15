import numpy as np
from skimage.io import imread
from skimage.transform import resize


def encode_rle(mask):
    """Returns encoded mask (run length) as a string.

    Parameters
    ----------
    mask : np.ndarray, 2d
        Mask that consists of 2 unique values: 0 - denotes background, 1 - denotes object.

    Returns
    -------
    str
        Encoded mask.

    Notes
    -----
    Mask should contains only 2 unique values, one of them must be 0, another value, that denotes
    object, could be different from 1 (for example 255).

    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


def decode_rle(rle_mask, shape=(320, 240)):
    """Decodes mask from rle string.

    Parameters
    ----------
    rle_mask : str
        Run length as string formatted.
    shape : tuple of 2 int, optional (default=(320, 240))
        Shape of the decoded image.

    Returns
    -------
    np.ndarray, 2d
        Mask that contains only 2 unique values: 0 - denotes background, 1 - denotes object.
    
    """
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for low, high in zip(starts, ends):
        img[low:high] = 1

    return img.reshape(shape)


def get_xy(images, img_path, img_mask_path=None, IMG_HEIGHT=240, IMG_WIDTH=240, IMG_CHANNELS=3):
    """Reads images and corresponding maks in data directories

    Parameters
    ----------
    images : list
        list containing the names of the images in the directory given by path `img_path`
    img_path : str
        path to directory of images    
    img_mask_path : str, optional (default=None)
        path to directory of masks 
    IMG_HEIGHT : int, optional (default=240)
        image height
    IMG_WIDTH : int, optional (default=240)
        image width
    IMG_CHANNELS : int, optional (default=3)
        image channels

    Returns
    -------
    tuple((np.ndarray, 4d),(np.ndarray, 4d))
        X_ np.array of np.array values of images read
        Y_ np.array of np.array values of image masks read
    
    """
    SIZE = len(images)
    
    X_= np.zeros((SIZE , IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_ = np.zeros((SIZE , IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for img_file_index in range(SIZE):

        ind = images[img_file_index].split(".")[0]
        
        img = imread(f"{img_path}/{ind}.jpg")[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        
        X_[img_file_index] = np.array(img)
        if(img_mask_path):
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
            
            mask_ = imread(f"{img_mask_path}/{ind}.png")
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        
            Y_[img_file_index] = np.maximum(mask, mask_)
        
    return (X_, Y_)


