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
    def __init__(self,path):
        self.filename = path

    def load_tfrecord(self,idx):
        dataset = tf.data.TFRecordDataset(self.filename, compression_type='')
        frame = None
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            break
        return frame
    
    def camera_image(self,frame):
        """return images from frame."""
        images = []
        for index, image in enumerate(frame.images):
            images.append(tf.image.decode_jpeg(image.image))
        return images
    
    def lidar(self,frame):
        (range_images, camera_projections,range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)
        points, _ = frame_utils.convert_range_image_to_point_cloud(frame,range_images, \
                                                                   camera_projections, \
                                                                   range_image_top_pose)
        points_all = np.concatenate(points, axis=0)
        return points_all

    def calib(self,frame):
        calibrations = frame.context.camera_calibrations
        Ks = []
        Es = []
        for idx, param in enumerate(calibrations):
            intrinsic = calibrations[idx].intrinsic
            K = np.array([[intrinsic[0],0,intrinsic[2]],\
                [0,intrinsic[1],intrinsic[3]],[0,0,1]])
            extrinsic = calibrations[idx].extrinsic.transform
            E = np.array(extrinsic).reshape((4,4))
            Ks.append(K)
            Es.append(E)
        return Ks,Es

    def get(self,idx):
        frame = self.load_tfrecord(self.filename,idx)
        images = self.camera_image(frame)
        lidar = self.lidar(frame)
        param = self.calib(frame)
        return images,lidar,param
