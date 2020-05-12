import open3d as o3d
import numpy as np
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

def colorize(pcd):
	pcd = [pcd] if type(pcd)!=list else pcd
	for p in pcd:
		p.paint_uniform_color(list(np.random.rand(3)))
if __name__ == '__main__':
	pcds = [get_point_cloud(read_npy('../experiments/data/waymo_lidar.npy')),\
		get_point_cloud(read_npy("../experiments/data/waymo_test.npy"))]
	colorize(pcds)
	vis(pcds)
