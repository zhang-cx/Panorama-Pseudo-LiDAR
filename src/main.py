import sys
sys.path.append('..')
import mono_api
import geometry
from dataset.waymo_dataset import *
from dataset.custom_dataset import *
from dataset.kitti_dataset import *
import open3d_utils as ovis
import kitti_utils
class PanoramaPL():
	def __init__(self,scale_factor=5.4,mode = 'kitti'):
		self.mode = mode
		self.images = []
		self.monodepth = mono_api.monodepth2()
		self.scale_factor = scale_factor

	def load_dataset(self,path):
		if self.mode == 'waymo':
			self.dataset = WaymoDataset(path)
		elif self.mode == 'custom':
			self.dataset = CustomDataset(path)
		elif self.mode == 'kitti':
			self.dataset = KittiDataset(path)
		else:
			raise NotImplementedError

	def single_point_cloud(self,image,K):
		if self.mode == 'kitti':
			disp = self.monodepth.inference(image)
			vole = kitti_utils.project_disp_to_depth(self.scale_factor/disp)
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


	# def run(self,idx):
	# 	images,lidar,calib = self.dataset.get(idx)
	# 	self.single_point_cloud(images,)


if __name__ == '__main__':
	test = 'Waymo'
	if test == 'KITTI':
		PL = PanoramaPL()
		PL.load_dataset('/home/ubuntu/KITTI')
		images,lidar,calib = PL.dataset.get('000004')
		sparse_vole = PL.single_point_cloud(images[0],calib)
		np.save('kitti000004.npy',sparse_vole)	
	if test == 'Waymo':
		timestamp = int(round(time.time()*1000))
		timestamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(timestamp/1000))
		dir_path = '../experiments/'+test+timestamp
		os.makedir(dir_path)
		PL = PanoramaPL(scale_facor = 1.0,mode = 'waymo')
		PL.load_dataset('/home/ubuntu/waymo/training_0000/segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord')
		images,lidar,calib = PL.dataset.get(30)
		sparse_vole = PL.fusion(images,calib[1],calib[0])
		np.save(os.path.join(dir_path,'waymo_test.npy'),sparse_vole)
		np.save(os.path.join(dir_path,'waymo_lidar.npy'),lidar)
		for idx,img in enumerate(images):
			cv2.imwrite(os.path.join(dir_path,'%d.npy'%idx),img)

