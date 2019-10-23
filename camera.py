#coding=utf-8
import cv2
import numpy as np
import mvsdk
import time
import threading
class Camera:
	def __init__(self):
		DevList = mvsdk.CameraEnumerateDevice()
		nDev = len(DevList)
		if nDev < 1:
			print("No camera was found!")
			return
		for i, DevInfo in enumerate(DevList):
			print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
		i = 0 if nDev == 1 else int(input("Select camera: "))
		DevInfo = DevList[0]
		
		# 打开相机
		self.hCamera = 0
		try:
			self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
		except mvsdk.CameraException as e:
			print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
			return
		# 获取相机特性描述
		self.cap = mvsdk.CameraGetCapability(self.hCamera)
		# 判断是黑白相机还是彩色相机
		monoCamera = (self.cap.sIspCapacity.bMonoSensor != 0)

		# 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
		if monoCamera:
			mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
		else:
			mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
		# 相机模式切换成连续采集
		mvsdk.CameraSetTriggerMode(self.hCamera, 0)
		# 手动曝光，曝光时间10ms
		mvsdk.CameraSetAeState(self.hCamera, 0)
		mvsdk.CameraSetExposureTime(self.hCamera, 10 * 1000)
		mvsdk.CameraSetAnalogGain(self.hCamera, 64)
		mvsdk.CameraSetGain(self.hCamera, 100, 110, 100)
		#mvsdk.CameraSetLutMode(self.hCamera, LUTMODE_PRESET);
		# 让SDK内部取图线程开始工作
		mvsdk.CameraPlay(self.hCamera)
		# 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
		self.FrameBufferSize = self.cap.sResolutionRange.iWidthMax * self.cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
		# 分配RGB buffer，用来存放ISP输出的图像
		# 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
		self.pFrameBuffer = mvsdk.CameraAlignMalloc(self.FrameBufferSize, 16)
		self.src = None
		self.state = True
		self.timestamp = None
		t = threading.Thread(target=self.read)
		t.setDaemon(True)		
		t.start()		

	def read(self):
		while True:
			try:
				self.timestamp = time.time()
				pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
				mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
				mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
				# 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
				# 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
				frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
				frame = np.frombuffer(frame_data, dtype=np.uint8)
				frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )
				frame = cv2.resize(frame, (640,480), interpolation = cv2.INTER_LINEAR)
			
				self.src = frame
				self.state = True
			except mvsdk.CameraException as e:
				if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
					print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )
				self.state = False
				return
	
	
	def __del__(self):
		cv2.destroyAllWindows()
		mvsdk.CameraUnInit(self.hCamera)
		mvsdk.CameraAlignFree(self.pFrameBuffer)
		
if __name__ == "__main__":
	camera = Camera() 
	state, frame = camera.read()
	
	while state and (cv2.waitKey(1) & 0xFF) != ord('q'):
		begain = time.time()
		state, frame = camera.read()
		cv2.imshow("Press q to end", frame)
		total = time.time() - begain
		print("fps", 1/total)

