import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
root_dir = os.getcwd()
dataset_dir = os.path.join(root_dir,'dataset')

name = input("Enter your name: ")
input_directory = os.path.join(dataset_dir,name)
if not os.path.exists(input_directory):
    os.makedirs(input_directory, exist_ok = 'True')
    count = 1
    print("[INFO] starting video stream...")
    video_capture = cv2.VideoCapture(0)
    while count <= 50:
        try:
            check, frame = video_capture.read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:  
                face = frame[y-5:y+h+5,x-5:x+w+5]
                resized_face = cv2.resize(face,(160,160))
                cv2.imwrite(os.path.join(input_directory,name + str(count) + '.jpg'),resized_face)
                cv2.rectangle(frame, (x,y), (x+w, y+h),(0,0,255), 2)
                count += 1
            # show the output frame
            cv2.imshow("Frame",frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        except Exception as e:
            pass
    video_capture.release()
    cv2.destroyAllWindows()
else:
    print("Photo already added for this user..Please provide another name for storing datasets")