import pyrealsense2 as rs
import numpy as np
import cv2
import keyboard
import realsense_device_manager as rsdm
import math as m
import geometry
import time
import open3d_utils as ovis
import os
class Multi_RS_Camera():
	def __init__(self):
		# self.cfg = rs.config()
		# self.cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
		# self.cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
		# self.cfg.enable_stream(rs.stream.pose)
		# self.device_manager = rsdm.DeviceManager(rs.context(),self.cfg)
		# self.device_manager.enable_all_devices()
		# self.device_manager.enable_emitter(True)
		#self.t265 = T265()
		self.d453 = D435()
		
	# def __del__(self):
	# 	self.device_manager.disable_streams()

	def capture(self):
		# frames = self.device_manager.poll_frames()
		# self.device_extrinsics = self.device_manager.get_depth_to_color_extrinsics(frames)
		# self.device_intrinsics = self.device_manager.get_device_intrinsics(frames)
		# print('success')
		depth,color,param = self.d453.capture()
		#T = self.t265.capture()
		return depth,color,param,None

	def extract_K(self,intrinsics):
		return np.array([[intrinsics.fx,0.0,intrinsics.ppx],\
						[0.0, intrinsics.fy,intrinsics.ppy],\
						[0.0, 0.0,1.0]])

	def run(self):
		depths = []
		images = []
		params = []
		transforms = []
		print('Start to enter command:')
		while True:
			cmd = input()
			timestamp = int(round(time.time()*1000))
			timestamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(timestamp/1000))
			if cmd == 'q':
				break
			if cmd == 's':
				depth,image,param,T = self.capture()
				depths.append(depth)
				images.append(image)
				params.append(param)
				transforms.append(T)
				
				print('[Success][%s] Stream captured'%timestamp)
			else:
				print('[Warning][%s] Wrong command'%timestamp)
		return depths,images,params,transforms

	def fusion(self,depths,params,transforms):
		assert len(depths) == len(params) == len(transforms), "Length is not correct"
		N = len(depths)
		pointclouds = []
		for i in range(N):
			rect = geometry.depth_to_rect(depths[i])
			K = self.extract_K(params[i][0])
			vole = geometry.rect_to_vole(rect,K)
			#vole = geometry.channel_exchange(vole)
			vole = geometry.rigid_transform(vole,np.linalg.inv(transforms[i]))
			pointclouds.append(vole)
		return np.concatenate(pointclouds,axis = 0),pointclouds

	def visualize(self,pc):
		print(pc.shape)
		ovis.vis_npy(pc)

	def vis_separate(self,pcs):
		pcds = [ovis.get_point_cloud(p) for p in pcs]
		ovis.colorize(pcds)
		ovis.vis(pcds)

class D435():
	def __init__(self):
		self.pipe = rs.pipeline()
		self.cfg = rs.config()
		#self.cfg.enable_stream(rs.stream.pose)
		self.cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
		self.cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
		profile = self.pipe.start(self.cfg)
		depth_sensor = profile.get_device().first_depth_sensor()
		self.depth_scale = depth_sensor.get_depth_scale()
		self.align = rs.align(rs.stream.color)



	def capture(self):
		#clipping_distance_in_meters = 100
		#clipping_distance = clipping_distance_in_meters /self.depth_scale
		while True:
			frames = self.pipe.wait_for_frames()
			aligned_frames = self.align.process(frames)
			aligned_depth_frame = aligned_frames.get_depth_frame()
			color_frame = aligned_frames.get_color_frame()
			intrinsics = aligned_depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
			extrinsics = aligned_depth_frame.get_profile().as_video_stream_profile().get_extrinsics_to(color_frame.get_profile())
			param = (intrinsics,extrinsics)
			if not aligned_depth_frame or not color_frame:
				continue
			depth_image = np.asanyarray(aligned_depth_frame.get_data())
			color_image = np.asanyarray(color_frame.get_data())
			print(intrinsics)
			depth = depth_image.astype(np.float32) * self.depth_scale
			#grey_color = 153
			#depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
			#bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
			break
		return depth,color_image,param


class T265():
	def __init__(self):
		self.pipe = rs.pipeline()
		self.cfg = rs.config()
		self.cfg.enable_stream(rs.stream.pose)
		profile = self.pipe.start(self.cfg)


	def capture(self):
		while True:
			frames = self.pipe.wait_for_frames()
			pose = frames.get_pose_frame()
			if not pose:
				continue
			data = pose.get_pose_data()
			R = geometry.quaternion2rotation((data.rotation.w,
										 data.rotation.x,
										 data.rotation.y,
										 data.rotation.z))
			print(R@R.T)
			t = data.translation
			t = np.array([t.x,t.y,t.z]).reshape((1,3))
			T = np.zeros((4,4))
			T[:3,:3] = R
			T[:3,3] = t
			T[3,3] = 1.0
			break
		return T

if __name__ == '__main__':
	# rs_cameras1 = D435()
	# rs_cameras2 = T265()
	# _,color = rs_cameras1.capture()
	# print(color)
	# transform = rs_cameras2.capture()
	# print(transform)
	# while True:   
	# 	if input():  
	# 		_,color = rs_cameras.D435()
	# 		#color = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
	# 		cv2.imwrite('test.png',color)
	# rs_cams = Multi_RS_Camera()
	# rs_cams.capture()
	cams = Multi_RS_Camera()
	depths,images,params,transforms = cams.run()
	# pc,pcs = cams.fusion(depths,params,transforms)
	# cams.vis_separate(pcs)
	timestamp = int(round(time.time()*1000))
	timestamp = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(timestamp/1000))
	os.makedirs(timestamp)
	for i in range(len(depths)):
		np.save('%s/depth_%d.npy'%(timestamp,i),depths[i])
		print(images[i])
		print(type(images[i]))
		cv2.imwrite('%s/image_%d.png'%(timestamp,i),images[i])
