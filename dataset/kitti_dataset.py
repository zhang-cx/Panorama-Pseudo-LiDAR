import sys
sys.append('..')
import src.kitti_utils as kitti_utils
import numpy as np 
import cv2
class KittiDataset():
	def __init__(self,path):
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
		self.kitt_path  = path

	def load_image(self,idx,mode):
		"""
		param: idx(str)
		param: mode(str): "training" or "testing"
		"""
		image_left = cv2.imread(os.path.join(self.kitt_path,mode,'image_2',str(idx)+'.png'))
		image_right = cv2.imread(os.path.join(self.kitt_path,mode,'image_3',str(idx)+'.png'))
		return [image_left,image_right]

	def calib(self,idx,mode):
		return kitti_utils.Calibration(os.path.join(self.kitt_path,mode,'calib',str(idx)+'.txt'))

	def lidar(self,idx,mode):
		velodyne = np.fromfile(os.path.join(self.kitt_path,mode,'velodyne',str(idx)+'.bin'),\
			 dtype=np.float32, count=-1).reshape([-1,4])
		return velodyne[:,:3]

	def get(self,idx,mode):
		images = self.load_image(idx,mode)
		lidar = self.lidar(idx,mode)
		calib = self.calib(idx,mode)
		return images,lidar,calib
