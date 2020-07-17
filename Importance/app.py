#from flask import Flask, render_template
from __future__ import print_function
from flask import Flask, render_template, Response
import cv2
import os
import time
import glob
# import main
from flask_socketio import SocketIO
from flask_apscheduler import APScheduler
import datetime
import mysql.connector
import requests 
import json

from flask_socketio import SocketIO, emit
from flask import Flask, render_template, url_for, copy_current_request_context
from random import random
from time import sleep
from threading import Thread, Event
# ====

from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import argparse
import sys
import json
import numpy as np
import pandas as pd
import os
import threading
from mtcnn.mtcnn import MTCNN

# ====
import math



# from flask import Flask, render_template
# from flask_socketio import SocketIO

__author__ = 'slynn'

# app = Flask(__name__)
# app = Flask(__name__, static_folder='C:/Users/lnkngoc/Desktop/FACE/display/')
app = Flask(__name__, static_folder='C:\\Users\\Administrator\\Documents\\FACE\\web_face_image\\')

app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

#turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)

#random number Generator Thread
thread = Thread()
thread_stop_event = Event()

# app = Flask(__name__, static_folder='C:\\Users\\Administrator\\Documents\\FACE\\web_face\\')
path_check = "./templates/display_image/img/images/"
# path_check = "C:\\Users\\Administrator\\Documents\\FACE\\web_face\\templates\\display_image\\img\\images"
count = 0
socketio = SocketIO(app)

# ============= Face ===============
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, help="Run camera recognition", default="camera")
args = parser.parse_args(sys.argv[1:]);
FRGraph = FaceRecGraph();
aligner = AlignCustom();
extract_feature = FaceFeature(FRGraph)
face_detect = MTCNNDetect(FRGraph, scale_factor=2); #scale_factor, rescales image for faster detection
xl = pd.ExcelFile('information.xls')
df = pd.read_excel(xl, 0, header=None,encoding='utf-8')
detector = MTCNN()
# ====
# detector = MTCNN()
upper_left = (300, 200)
bottom_right = (1400, 700)
@app.route('/')
def index():
    #only by sending this page first will the client be connected to the socketio instance
    return render_template('index.html')

@app.route('/camera',  methods=['GET', 'POST'])
def camera():
    return render_template('camera.html')

@socketio.on('connect', namespace='/test')
def test_connect():
    # need visibility of the global thread object
    global thread
    print('Client connected')

    #Start the random number generator thread only if the thread has not been started before.
    if not thread.isAlive():
        print("Starting Thread")
        # thread = socketio.start_background_task(randomNumberGenerator)
        thread = socketio.start_background_task(show_image)

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

# ====================================[ start show_image-----------------------------
file_name = []
file_name_list = []
latest_file_list = []
len_old = 0
def show_image():
    # print(text, str(datetime.datetime.now()))
    # import glob
    # import os
    global latest_file_list
    global len_old
    global file_name
    global file_name_list
    while not thread_stop_event.isSet():
        list_of_files = glob.glob( "./templates/display_image/img/image/*") # * means all if need specific format then *.csv
        # path = "C:/Users/lnkngoc/Desktop/FACE/web_face/templates/display_image/img/image/"
        # list_of_files = os.listdir(os.path.expanduser(path))

        latest_file = max(list_of_files, key=os.path.getctime) # Latest file created is full name with directory
        file_name = latest_file[36:]  # Latest file created is file name without directory
        len_new = len(list_of_files)
        if len_old < len_new:
            latest_file_list.append(latest_file)
            file_name_list.append(file_name)
            len_old = len_new
        
        len_latest_file_list = len(latest_file_list)
        show1 = latest_file_list[len_latest_file_list-8:] # List file name with directory

        len_file_name = len(file_name)
        show2 = file_name_list[len_file_name-8:] # List file name without directory
        # socketio.emit('my response', latest_file)
        # print('Danh sach',list_of_files)
        # print ('latest_file',latest_file)
        print ('file_name',file_name)
        # print ('show',show)
        # return latest_file
        socketio.emit('new_image', {'image': file_name}, namespace='/test')
        socketio.sleep(1)
# show_image()
# ------------------------------end show_image================================================]

# ============================= start Camera==================================================[
def find_camera(id):
    # cameras = ['rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp',
    # 'rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp']
    ## cameras = [0,1,"rtsp://192.168.86.39:554/user=admin&password=&channel=1&stream=0.sdp?"]
    cameras = ["rtsp://test:12345678x@X@192.168.85.76:554/Streaming/Channels/101",
    "rtsp://test:12345678x@X@192.168.85.75:554/Streaming/Channels/101",
    "rtsp://test:12345678x@X@192.168.85.48:554/Streaming/Channels/101",
    "rtsp://test:12345678x@X@192.168.85.20:554/Streaming/Channels/101"]
    return cameras[int(id)]
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
#  for webcam use zero(0)
 


# =========== gen_frames==============
def gen_frames(camera_id):
     
    cam = find_camera(camera_id)
    cap=  cv2.VideoCapture(cam)
    
    while True:
        # for cap in caps:
        # # Capture frame-by-frame
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

# =============== video_feed =============
@app.route('/video_feed/<string:id>/', methods=["GET"])
def video_feed(id):
   
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ============================= end Canera========================================================]
# # ============================ start Face old=====================================================[

 
def capture(path, name, frame_detect):
    cv2.imwrite(os.path.join(path, '%s.jpg'%name), frame_detect)

# def face_frames(camera_id):  
#     cam = find_camera(camera_id)
#     cap=  cv2.VideoCapture(cam)
#     global count
#     global path_check
#     upper_left = (250, 100)
#     bottom_right = (700, 700)
#     while True:
#         success, frame = cap.read()  # read the camera frame
#         # frame = ConvertToYUYV(frame)
#         if not success:
#             cap=  cv2.VideoCapture(cam)
#         else:

#             # frame = cv2.resize(frame, (1280,720))
#             # frame = cv2.resize(frame, (426,240))
#             if count%10 == 0:
#                 count = 0
#                 frame_process =  frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
#                 rects, landmarks = face_detect.detect_face(frame_process,30);#min face size is set to 20x20
#                 for (i, rect) in enumerate(rects):
#                     # detect face 
#                     face_image = frame_process.copy()
#                     # face_image = face_image[rect[1]:rect[1]+rect[3]+50, rect[0]-40:rect[0]+rect[2]+40]
#                     # capture image
#                     day = time.strftime("%H-%M-%S-%d-%m-%Y")
#                     t1 = threading.Thread(target=capture, args=(path_check, day, face_image,))
#                     t1.start()
#                     cv2.rectangle(frame_process, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]),(255,0,0),2)

               
#                 frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]] = frame_process
#                 cv2.rectangle(frame, (upper_left[0], upper_left[1]), (upper_left[0]+bottom_right[0], upper_left[1]+bottom_right[1]),(255,0,0),2)


#                 # cv2.imshow("Frame",frame)
                
#                 ################ Old source #######################
              
#                 ################ Old source #######################
#                 ret, buffer = cv2.imencode('.jpg', frame)
#                 frame = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
#             else:
#                 cv2.imshow('frame',frame)
#                 count = count + 1
            
# # ============================ end Face old===============================================================]
# =============== video_face =============
@app.route('/video_face/<string:id>/', methods=["GET"])
def video_face(id):
   
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(face_frames(id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================ start Face+ Camera new =============================================================-----------------------------------------------------------------[
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
			global path_check
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
							bounding_box[0], bounding_box[1] = abs(bounding_box[0]), abs(bounding_box[1])
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

def face_frames(camera_id):  
    cam = find_camera(camera_id)
    # cap=  cv2.VideoCapture(cam)
    # stream_link = "rtsp://test:12345678x@X@192.168.85.75:554/Streaming/Channels/101"
    video_stream_widget = VideoStreamWidget(cam)
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass
    ret, buffer = cv2.imencode('.jpg', frame)
#                 frame = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        
# ============================ end Face+ Camera new ===============================================================-----------------------------------------------------------------------]





if __name__ == '__main__':
	# app.run()
	# app.run(host='0.0.0.0',port=80)
	# app.debug = True
	# socketio.run(app, debug=True, host='0.0.0.0', port=80)
	# scheduler = APScheduler()
	# scheduler.add_job(func=show_image, args=['call show_image'], trigger='interval', id='show_image', seconds=1)
	# scheduler.start()

	socketio.run(app, debug=True, host='0.0.0.0', port=80)
	# socketio.run(app, debug = True, use_reloader = False, port=1111)



