import pyrealsense2 as rs
import numpy as np
import cv2
import keyboard
import realsense_device_manager as rsdm
import math as m
import geometry
class Multi_RS_Camera():
	def __init__(self):
		self.cfg = rs.config()
		self.cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
		self.cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
		self.cfg.enable_stream(rs.stream.pose)
		self.device_manager = rsdm.DeviceManager(rs.context(),self.cfg)
		self.device_manager.enable_all_devices()
		#self.device_manager.enable_emitter(True)
		
	def __del__(self):
		self.device_manager.disable_streams()

	def capture(self):
		frames = self.device_manager.poll_frames()
		self.device_extrinsics = self.device_manager.get_depth_to_color_extrinsics(frames)
		self.device_intrinsics = self.device_manager.get_device_intrinsics(frames)
		print('success')


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

	def __del__(self):
		self.pipe.stop()


	def capture(self):
		#clipping_distance_in_meters = 100
		#clipping_distance = clipping_distance_in_meters /self.depth_scale
		while True:
			frames = self.pipe.wait_for_frames()
			aligned_frames = self.align.process(frames)
			aligned_depth_frame = aligned_frames.get_depth_frame()
			color_frame = aligned_frames.get_color_frame()
			print(aligned_depth_frame.get_profile().as_video_stream_profile().get_intrinsics())
			print(aligned_depth_frame.get_profile().as_video_stream_profile().get_extrinsics_to(color_frame.get_profile()))
			
			if not aligned_depth_frame or not color_frame:
				continue
			depth_image = np.asanyarray(aligned_depth_frame.get_data())
			color_image = np.asanyarray(color_frame.get_data())
			depth = depth_image.astype(np.float32) * self.depth_scale
			#grey_color = 153
			#depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
			#bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
			break
		return depth,color_image


class T265():
	def __init__(self):
		self.pipe = rs.pipeline()
		self.cfg = rs.config()
		profile = self.pipe.start(self.cfg)


	def __del__(self):
		self.pipe.stop()


	def capture(self):
		while True:
			frames = self.pipe.wait_for_frames()
			pose = frames.get_pose_frame()
			if not pose:
				continue
			data = pose.get_pose_data()
			R = geometry.quaternion2rotation(data.rotation.w,
										 data.rotation.x,
										 data.rotation.y,
										 data.rotation.z)
			
			t = data.translation
			t = np.array([t.x,t.y,t.z]).reshape((1,3))
			T = np.zeros((4,4))
			T[:3,:3] = R
			T[:3,3] = t
			T[3,3] = 1.0
			break
		return T

if __name__ == '__main__':
	rs_cameras1 = D435()
	#rs_cameras2 = T265()
	_,color = rs_cameras1.capture()
	#print(color)
	#transform = rs_cameras2.capture()
	#print(transform)
	# while True:   
	# 	if input():  
	# 		_,color = rs_cameras.D435()
	# 		#color = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
	# 		cv2.imwrite('test.png',color)
	# rs_cams = Multi_RS_Camera()
	# rs_cams.capture()