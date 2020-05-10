import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

class WaymoDataset():
    def __init__():
        pass
    def load_tfrecord(self,filename,idx):
        dataset = tf.data.TFRecordDataset(filename, compression_type='')
        frame = None
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            break
        return frame
    
    def camera_image(frame):
        """return images from frame."""
        images = []
        for index, image in enumerate(frame.images):
            images.append(tf.image.decode_jpeg(image.image))
        return images
    
    def lidar(frame):
        pass


