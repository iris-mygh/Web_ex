import cv2
from mtcnn.mtcnn import MTCNN
import threading
import time
import os
import math
 



path_check = "C:\\Users\\Administrator\\Documents\FACE\\web_face\\templates\\display_image\\img\\images"
detector = MTCNN()
cap = cv2.VideoCapture("rtsp://test:12345678x@X@192.168.85.75:554/Streaming/Channels/101")
count = 0
# upper_left = (300, 200)
# bottom_right = (1000, 700)
upper_left = (300, 200)
bottom_right = (1400, 700)
def capture(path, name, frame_detect):
	cv2.imwrite(os.path.join(path, '%s.jpg'%name), frame_detect)


while True: 
    #Capture frame-by-frame
    flag, frame = cap.read()
    #Use MTCNN to detect faces
    # print(cap.get(5))
    if flag == True:
        # frame = cv2.resize(frame, (2*680,2*480))
        if count%10 == 0:
            # frame1 = cv2.resize(frame1, (2*680,2*480))
            frame =  frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
            count = 0
            result = detector.detect_faces(frame)
            if len(result) > 0:
                for person in result:
                    bounding_box = person['box']
                    cv2.rectangle(frame,
                                (bounding_box[0], bounding_box[1]),
                                (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                                (0,155,255),
                                2)
                    # day = time.strftime("%H-%M-%S-%d-%m-%Y")
                    # t1 = threading.Thread(target=capture, args=(path_check, day, frame,))
                    # t1.start()
            # frame1[upper_left[1] : bottom_right[1], upper_left[0] :q bottom_right[0]] = frame
            # cv2.rectangle(frame1, (upper_left[0], upper_left[1]), (upper_left[0]+bottom_right[0], upper_left[1]+bottom_right[1]),(255,0,0),2)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break
        count = count + 1
    else:
        cap = cv2.VideoCapture("rtsp://test:12345678x@X@192.168.85.75:554/Streaming/Channels/101")
    
    # print(count)
    #display resulting frame

#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()