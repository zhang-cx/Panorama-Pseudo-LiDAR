import numpy as np
import os
import cv2

class CustomDataset():
	def __init__(self,path):
		self.path = path

	def load_images(self):
		images_name = os.listdir(self.path)
		images = []
		for idx,name in enumerate(images_name):
			images.append(cv2.imread(os.join(self.path,name)))
		return images

	def calib(self,images):
		pass


	def get(self):
		images = self.load_images(self.path)
		param = self.calib(images)
		return images,param