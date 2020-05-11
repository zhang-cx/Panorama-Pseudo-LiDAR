import sys
sys.path.append('..')
import mono_api
import geometry
from dataset.waymo_dataset import *
from dataset.custom_dataset import *
from dataset.kitti_dataset import *
import open3d_utils as ovis

class PanoramaPL():
	def __init__(self,scale_factor=5.4,mode = 'kitti'):
		self.mode = mode
		self.images = []
		self.monodepth = mono_api.monodepth2()
		self.scale_factor = scale_factor

	def load_dataset(self,name):
		if name == 'waymo':
			self.dataset = WaymoDataset()
		elif name == 'custom':
			self.dataset = CustomDataset()
		elif name == 'kitti':
			self.dataset = KittiDataset()
		else:
			raise NotImplementedError

	def single_point_cloud(self,image,K):
		if self.mode == 'kitti':
			disp = self.monodepth.inference(image)
			rect = geometry.disp_to_rect(disp,self.scale_factor)
			vole = K.project_image_to_velo(rect)
			sparse_vole = geometry.gen_sparse_points(vole)
		else:
			disp = self.monodepth.inference(image)
			rect = geometry.disp_to_rect(disp,self.scale_factor)
			vole = geometry.rect_to_vole(rect,K)
			vole = geometry.channel_exchange(vole)
			sparse_vole = geometry.gen_sparse_points(vole)
		return sparse_vole

	def fusion(self,images,Es,Ks):
		mpoints = []
		for idx, img in enumerate(images):
			points = self.single_point_cloud(img,Ks[idx])
			points = geometry.rigid_transform(points,Es[idx])
			mpoints.append(points)
		mpoints = np.concatenate(mpoints,axis=0)
		return mpoints

	def visualize(self,points):
		ovis.vis_npy(points)
		
	def detection(self,points):
		pass




if __name__ == '__main__':
	PL = PanoramaPL()
