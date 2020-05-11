import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
#import pcl.pcl_visualization
#import kitti_util
#import argparse
import os

import numpy as np
import tqdm
import open3dvis


def pcl_vis_color(points):
    points = points.astype(np.float32)
    cloud = pcl.PointCloud_PointXYZRGB(points)
    visual = pcl.pcl_visualization.CloudViewing()
    visual.ShowColorCloud(cloud)
    flag = True
    while flag:
        flag != visual.WasStopped()

def pcl_vis_white(points):
    points = points.astype(np.float32)
    cloud = pcl.PointCloud(points)
    visual = pcl.pcl_visualization.CloudViewing()
    visual.ShowMonochromeCloud(cloud)
    flag = True
    while flag:
        flag != visual.WasStopped()

def read_file(file_path):
    return np.load(file_path)



def colorize(points,color):
    color = np.array([color]).reshape((1,1))
    pnum = points.shape[0]
    return np.concatenate([points,np.repeat(color,pnum,axis=0)],axis=1)




colors = [12000,255*255*255,16000,8190,4330]
_,points = merge(colors,4.)
fake = [open3dvis.get_point_cloud(p) for p in points]
gdt = open3dvis.get_point_cloud(np.load('all.npy'))
open3dvis.colorize(fake+[gdt])
open3dvis.vis(fake+[gdt])

