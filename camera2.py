#coding=utf-8
import cv2
import numpy as np
import mvsdk
import time
import threading



class Camera2:
	def __init__(self):
		DevList = mvsdk.CameraEnumerateDevice()
		nDev = len(DevList)
		if nDev < 1:
			print("No camera was found!")
			return
		for i, DevInfo in enumerate(DevList):
			print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
		
		DevInfo0 = DevList[0]
		DevInfo1 = DevList[1]
		
		# 打开相机
		self.hCamera0 = 0
		self.hCamera1 = 0
		try:
			self.hCamera0 = mvsdk.CameraInit(DevInfo0, -1, -1)
			self.hCamera1 = mvsdk.CameraInit(DevInfo1, -1, -1)
			print("init successfully", self.hCamera0, self.hCamera1)
		except mvsdk.CameraException as e:
			print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
			return
		# 获取相机特性描述
		self.cap0 = mvsdk.CameraGetCapability(self.hCamera0)
		self.cap1 = mvsdk.CameraGetCapability(self.hCamera1)
		# 判断是黑白相机还是彩色相机
		monoCamera0 = (self.cap0.sIspCapacity.bMonoSensor != 0)
		monoCamera1 = (self.cap1.sIspCapacity.bMonoSensor != 0)
		# 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
		if monoCamera0:
			mvsdk.CameraSetIspOutFormat(self.hCamera0, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
		else:
			mvsdk.CameraSetIspOutFormat(self.hCamera0, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
		if monoCamera1:
			mvsdk.CameraSetIspOutFormat(self.hCamera1, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
		else:
			mvsdk.CameraSetIspOutFormat(self.hCamera1, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
		# 相机模式切换成连续采集
		mvsdk.CameraSetTriggerMode(self.hCamera0, 0)
		mvsdk.CameraSetTriggerMode(self.hCamera1, 0)
		# 手动曝光，曝光时间10ms
		mvsdk.CameraSetAeState(self.hCamera0, 0)
		mvsdk.CameraSetExposureTime(self.hCamera0, 10 * 1000)
		mvsdk.CameraSetAnalogGain(self.hCamera0, 20)
		mvsdk.CameraSetGain(self.hCamera0, 100, 131, 110)

		mvsdk.CameraSetAeState(self.hCamera1, 0)
		mvsdk.CameraSetExposureTime(self.hCamera1, 10 * 1000)
		mvsdk.CameraSetAnalogGain(self.hCamera1, 20)
		mvsdk.CameraSetGain(self.hCamera1, 100, 131, 110)
		#mvsdk.CameraSetLutMode(self.hCamera, LUTMODE_PRESET);
		# 让SDK内部取图线程开始工作
		mvsdk.CameraPlay(self.hCamera0)
		mvsdk.CameraPlay(self.hCamera1)
		# 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
		self.FrameBufferSize0 = self.cap0.sResolutionRange.iWidthMax * self.cap0.sResolutionRange.iHeightMax * (1 if monoCamera0 else 3)
		self.FrameBufferSize1 = self.cap1.sResolutionRange.iWidthMax * self.cap1.sResolutionRange.iHeightMax * (1 if monoCamera0 else 3)
		# 分配RGB buffer，用来存放ISP输出的图像
		# 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
		self.pFrameBuffer0 = mvsdk.CameraAlignMalloc(self.FrameBufferSize0, 16)
		self.pFrameBuffer1 = mvsdk.CameraAlignMalloc(self.FrameBufferSize1, 16)
				
		self.state0 = False
		self.state1 = False
		self.frame0 = None
		self.frame1 = None
		print("init finish")

	def read(self):
		self.read_left()
		self.read_right()
		#t1 = threading.Thread(target=self.read_left)
		#t2 = threading.Thread(target=self.read_right)
		#t1.start()
		#t2.start()
		#t1.join()
		#t2.join()
		#print(self.state0, self.state1)
		if self.state0 and self.state1:
			return True, self.frame0, self.frame1
		else:
			return False, None, None

	def read_left(self):
		try:
			
			pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera0, 200)
			mvsdk.CameraImageProcess(self.hCamera0, pRawData, self.pFrameBuffer0, FrameHead)
			mvsdk.CameraReleaseImageBuffer(self.hCamera0, pRawData)
			# 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
			# 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
			
			frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer0)
			frame = np.frombuffer(frame_data, dtype=np.uint8)
			frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )
			frame = cv2.resize(frame, (640,480), interpolation = cv2.INTER_LINEAR)
			self.frame0 = frame
			self.state0 = True
			
		except mvsdk.CameraException as e:
			self.state0 = False
			if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
				print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )
			
	def read_right(self):
		try:
			pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera1, 200)
			mvsdk.CameraImageProcess(self.hCamera1, pRawData, self.pFrameBuffer1, FrameHead)
			mvsdk.CameraReleaseImageBuffer(self.hCamera1, pRawData)
			# 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
			# 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
			frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer1)
			frame = np.frombuffer(frame_data, dtype=np.uint8)
			frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )
			frame = cv2.resize(frame, (640,480), interpolation = cv2.INTER_LINEAR)
			self.frame1 = frame
			self.state1 = True
		except mvsdk.CameraException as e:
			self.state1 = False
			if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
				print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )
	
	
	def __del__(self):
		print("in __del__")
		cv2.destroyAllWindows()
		mvsdk.CameraUnInit(self.hCamera0)
		mvsdk.CameraAlignFree(self.pFrameBuffer0)
		mvsdk.CameraUnInit(self.hCamera1)
		mvsdk.CameraAlignFree(self.pFrameBuffer1)
		
if __name__ == "__main__":
	camera = Camera2() 
	state, frame0, frame1 = camera.read()
	print(state)
	while state and (cv2.waitKey(1) & 0xFF) != ord('q'):
		begin = time.time()
		state, frame0, frame1 = camera.read()
		#cv2.imshow("Press q to end", frame0)
		#cv2.imshow("frame2", frame1)
		frame0 = frame0.copy()
		frame1 = frame1.copy()
		print("fps", 1/(time.time() - begin))

