import cv2


class Tracker:
	def __init__(self, tracker_name):
		#self.OPENCV_OBJECT_TRACKERS = {
		#	"csrt": cv2.TrackerCSRT_create,
		#	"kcf": cv2.TrackerKCF_create,
		#	"boosting": cv2.TrackerBoosting_create,
		#	"mil": cv2.TrackerMIL_create,
		#	"tld": cv2.TrackerTLD_create,
		#	"medianflow": cv2.TrackerMedianFlow_create,
		#	"mosse": cv2.TrackerMOSSE_create
		#	}
		
		self.tracker = cv2.TrackerKCF_create()

	def init(self, frame, box):
		self.tracker.init(frame, box)

	def update(self, frame):
		return self.tracker.update(frame)
		

if __name__ == '__main__':
	tracker = Tracker("kcf")
