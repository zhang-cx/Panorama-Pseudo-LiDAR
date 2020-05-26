import open3d as o3d
import numpy as np
import geometry
import cv2
def read_npy(path):
	return np.load(path)


def get_point_cloud(npy):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(npy)
	return pcd

def vis(pcd):
	pcd = [pcd] if type(pcd)!=list else pcd
	o3d.visualization.draw_geometries(pcd)

def vis_npy(npy):
	pcd = get_point_cloud(npy)
	vis(pcd)

def vis_npys(lst):
	pcds = []
	for npy in lst:
		pcd = get_point_cloud(npy)
		pcd.paint_uniform_color(list(np.random.rand(3)))
		pcds.append(pcd)
	vis(pcds)


def colorize(pcd):
	pcd = [pcd] if type(pcd)!=list else pcd
	for p in pcd:
		p.paint_uniform_color(list(np.random.rand(3)))

def vis_depth(depth):
	cv2.imwrite('depth.png',(depth*255).astype(np.uint8))

if __name__ == '__main__':
	name="Waymo2020-05-15-03-03-41"
	# pcds = [get_point_cloud(read_npy("../experiments/%s/waymo_lidar.npy"%name)),\
	# 	get_point_cloud(read_npy("../experiments/%s/waymo_test.npy"%name))]
	# colorize(pcds)
	# vis(pcds)
	for i in range(5):
		img = cv2.imread('../experiments/%s/%d.png'%(name,i))
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		cv2.imwrite('../experiments/%s/%d.png'%(name,i),img)

	# pc = np.load('/Users/vincent/Desktop/Panorama-Pseudo-LiDAR/data/2020-05-13-18-44-20/depth_13.npy')
	# pc = geometry.depth_to_rect(pc)
	# pc = pc[pc[:,2]!=0]
	# vis_npy(pc)	
	
	#Vis KITTI
	# idx=32
	# gdt = np.fromfile("/Users/vincent/Downloads/data_object_velodyne/training/velodyne/0000%d.bin"%idx, dtype=np.float32, count=-1).reshape([-1,4])[:,:3]
	# #predict = np.load("../data/kitti0000%d.npy"%idx)
	# predict1 = np.load("../data/kitti0000%d_1.npy"%idx)
	# vis_npys([gdt,predict1])

	#vis_depth(np.load('../data/mono_depth_2020_05_14/WechatIMG2953.npy'))

	# npy = read_npy('/Users/vincent/Dropbox/findjob/cs280/result_point_cloud.npy')
	# npy = geometry.channel_exchange(npy)
	# vis_npys([gdt,npy])
