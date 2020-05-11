import numpy as np
import os
import cv2
class CustomDataset():
	def __init__(self):
		pass

	def load_images(self,path):
		images_name = os.listdir(path)
		images = []
		for idx,name in enumerate(images_name):
			images.append(cv2.imread(os.join(path,name)))
		return images

	def calib(self,images):
		pass


	def get(self,path):
		images = self.load_images(path)
		param = self.calib(images)
		return images,param