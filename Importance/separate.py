from threading import Thread, Lock
import cv2
import queue

frame_return = queue.Queue()
# lock_thread = Lock()
class WebcamVideoStream:
	def __init__(self, src = 0, width = 320, height = 240) :
		self.stream = cv2.VideoCapture(src)
		# self.stream.set(cv2.CV_CAP_PROP_FRAME_WIDTH, width)
		# self.stream.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, height)
		(self.grabbed, self.frame) = self.stream.read()
		self.started = False
		self.read_lock = Lock()

	def start(self):
		if self.started :
			print("already started!!")
			return None
		self.started = True
		self.thread = Thread(target=self.update, args=())
		self.thread.start()
		return self

	def update(self):
		while self.started :
			(grabbed, frame) = self.stream.read()
			self.read_lock.acquire()
			self.grabbed, self.frame = grabbed, frame
			frame_return.put(self.frame)
			self.read_lock.release()

	def read(self):
		self.read_lock.acquire()
		frame = self.frame.copy()
		self.read_lock.release()
		return frame

	def stop(self):
		self.started = False
		self.thread.join()

	def __exit__(self, exc_type, exc_value, traceback) :
		self.stream.release()

# def process_camera(frame):
# 	lock_thread.acquire()
# 	up_left= [10, 20]
# 	bottom_right = [70, 70]
# 	cv2.rectangle(frame, up_left, bottom_right, (255,0,0),2)
# 	cv2.imshow("process", frame)
# 	lock_thread.release()

if __name__ == "__main__":
	vs = WebcamVideoStream().start()
	# thread()
	while True :
		# frame = vs.read()
		cv2.imshow('webcam', frame_return.get())
		# process = Thread(target = process_camera, args=(frame_return.get(),))
		# process.start()

		if cv2.waitKey(1) == 27:
			break

	vs.stop()
	cv2.destroyAllWindows()