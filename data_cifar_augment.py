# from https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

def flip(*args):
    """Flip augmentation
    Args:
        x: Image to flip
    Returns:
        Augmented image
    """
    x = args[0]
    args_ = list(args)
    if np.random.uniform(0, 1) > 0.75:

        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        args_[0] = x
    return args_

def color(*args):
    """Color augmentation
    Args:
        x: Image
    Returns:
        Augmented image
    """
    x = args[0]
    args_ = list(args)
    if np.random.uniform(0, 1) > 0.75:

        x = tf.image.random_hue(x, 0.08)
        x = tf.image.random_saturation(x, 0.6, 1.6)
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.7, 1.3)
        args_[0] = x
    return args_

def rotate(*args):
    """Rotation augmentation
    Args:
        x: Image
    Returns:
        Augmented image
    """
    x = args[0]
    args_ = list(args)
    if np.random.uniform(0, 1) > 0.75:
        args_[0] = tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    return args_

def zoom(*args):
    """Zoom augmentation
    Args:
        x: Image
    Returns:
        Augmented image
    """
    x = args[0]
    args_ = list(args)
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[int(np.floor(np.random.uniform(low=0, high=len(scales))))]

    if np.random.uniform(0, 1) > 0.85:
        args_[0] =  random_crop(x)
    # Only apply cropping 50% of the time
    return args_

def clip(*args):
    x = args[0]
    args_ = list(args)
    x = tf.clip_by_value(x, 0, 1)
    args_[0] = x
    return args_
