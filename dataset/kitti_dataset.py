import sys
sys.path.append('..')
import src.kitti_utils as kitti_utils
import numpy as np 
import cv2
import os
class KittiDataset():
	def __init__(self,path,mode = 'training'):
		"""
		KITTI
		- training
		  - image_2
		  - image_3
		  - label_2
		  - velodyne     
		  - calib 
		- testing 
		  - image_2
		  - image_3
		  - velodyne     
		  - calib 
		"""
		self.kitt_path = path
		self.mode = mode

	def load_image(self,idx):
		"""
		param: idx(str)
		param: mode(str): "training" or "testing"
		"""
		image_left = cv2.imread(os.path.join(self.kitt_path,self.mode,'image_2',str(idx)+'.png'))
		image_right = cv2.imread(os.path.join(self.kitt_path,self.mode,'image_3',str(idx)+'.png'))
		return [image_left,image_right]

	def calib(self,idx):
		return kitti_utils.Calibration(os.path.join(self.kitt_path,self.mode,'calib',str(idx)+'.txt'))

	def lidar(self,idx):
		velodyne = np.fromfile(os.path.join(self.kitt_path,self.mode,'velodyne',str(idx)+'.bin'),\
			 dtype=np.float32, count=-1).reshape([-1,4])
		return velodyne[:,:3]

	def get(self,idx):
		images = self.load_image(idx)
		lidar = self.lidar(idx)
		calib = self.calib(idx)
		return images,lidar,calib
