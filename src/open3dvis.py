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

vis(get_point_cloud(read_npy('all.npy')))
