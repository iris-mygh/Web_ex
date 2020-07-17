from flask import Flask, render_template, Response
import cv2
from flask_socketio import SocketIO
from flask_apscheduler import APScheduler

app = Flask(__name__)
app = Flask(__name__, static_folder='C:/Users/lnkngoc/Desktop/WEB/multiple-camera-stream-master/')
# socketio = SocketIO(app)

def find_camera(id):
    # cameras = ['rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp',
    # 'rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp']
    cameras = ["rtsp://192.168.86.39:554/user=admin&password=&channel=1&stream=0.sdp?",0,1]
    return cameras[int(id)]
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
#  for webcam use zero(0)
 

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


@app.route('/video_feed/<string:id>/', methods=["GET"])
def video_feed(id):
   
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    app.debug = True
    # scheduler = APScheduler()
    # scheduler.start()
    # socketio.run(app, debug=True)

