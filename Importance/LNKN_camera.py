from threading import Thread

import cv2
from mtcnn.mtcnn import MTCNN
import threading
import time
import os
import math
import numpy as np


path_check = "C:\\Users\\Administrator\\Documents\FACE\\web_face\\templates\\display_image\\img\\images"
detector = MTCNN()
upper_left = (300, 200)
bottom_right = (1400, 700)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
class VideoStreamWidget(object):
	def __init__(self, src=0):
		# Create a VideoCapture object
		self.capture = cv2.VideoCapture(src)
		self.count = 0
		# Start the thread to read frames from the video stream
		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = True
		self.thread.start()

	def update(self):
		# Read the next frame from the stream in a different thread
		while True:
			if self.capture.isOpened():
				(self.status, self.frame) = self.capture.read()

#------------------------------------------ Old--------------------------------------------[
	# def show_frame(self):
	#     # Display frames in main program
	#     if self.status:
	#         self.frame = self.maintain_aspect_ratio_resize(self.frame, width=600)
	#         cv2.imshow('IP Camera Video Streaming', self.frame)

	#     # Press Q on keyboard to stop recording
	#     key = cv2.waitKey(1)
	#     if key == ord('q'):							name = time.strftime("%H-%M-%S-%d-%m-%Y")
	#         self.capture.release()
	#         cv2.destroyAllWindows()
	#         exit(1)
#------------------------------------------ Old--------------------------------------------]

# #------------------------------------------ New- model--------------------------------------------[
	def show_frame(self):
			# Display frames in main program
			if self.status:
				# self.frame = self.maintain_aspect_ratio_resize(self.frame, width=600)
				if self.count == 10:
					self.frame1 =  self.frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
					self.count = 0
					result = detector.detect_faces(self.frame1)
					if len(result) > 0:
						for person in result:
							bounding_box = person['box']
							cv2.rectangle(self.frame1,
										(bounding_box[0], bounding_box[1]),
										(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
										(0,155,255),
										2)
							name = time.strftime("%H-%M-%S-%d-%m-%Y")
							cv2.imwrite(os.path.join(path_check, '%s.jpg'%name), self.frame1[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]])
					# out.write(self.frame1)
					cv2.imshow('IP Camera Video Streaming', self.frame1)
				self.count = self.count +1
				print(self.count)
			# Press Q on keyboard to stop recording
			key = cv2.waitKey(1)
			if key == ord('q'):
				self.capture.release()
				# out.release()
				cv2.destroyAllWindows()
				exit(1)
# #------------------------------------------ New- model--------------------------------------------]

	# Resizes a image and maintains aspect ratio
	def maintain_aspect_ratio_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
		# Grab the image size and initialize dimensions
		dim = None
		(h, w) = image.shape[:2]

		# Return original image if no need to resize
		if width is None and height is None:
			return image

		# We are resizing height if width is none
		if width is None:
			# Calculate the ratio of the height and construct the dimensions
			r = height / float(h)
			dim = (int(w * r), height)
		# We are resizing width if height is none
		else:
			# Calculate the ratio of the 0idth and construct the dimensions
			r = width / float(w)
			dim = (width, int(h * r))

		# Return the resized image
		return cv2.resize(image, dim, interpolation=inter)

if __name__ == '__main__':
	stream_link = "rtsp://test:12345678x@X@192.168.85.75:554/Streaming/Channels/101"
	video_stream_widget = VideoStreamWidget(stream_link)
	while True:
		try:
			video_stream_widget.show_frame()
		except AttributeError:
			pass