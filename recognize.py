import cv2
import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from mark_attendance import Mark_Attendance
from statistics import mode
from datetime import datetime
import pandas as pd


root_dir = os.getcwd()
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
model_path = os.path.join(root_dir,'models/facenet_keras.h5')
facenet_model = load_model(model_path)

embeddings_model_file = os.path.join(root_dir,"models/embeddings.pickle")
recognizer_model_file = os.path.join(root_dir,"models/recognizer.pickle")

def load_embeddings_and_labels():
    data = pickle.loads(open(embeddings_model_file, "rb").read())
    # encoding labels by names
    label = LabelEncoder()
    names = np.array(data["names"])                       
    labels = label.fit_transform(names)
    # getting names
    # getting embeddings
    Embeddings = np.array(data["embeddings"])
    return [label,labels,Embeddings,names]

def normalize_pixels(imagearrays):
	face_pixels = np.array(imagearrays)
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	return face_pixels


predictions = []
if os.path.exists(embeddings_model_file and recognizer_model_file): 
    [label,labels,Embeddings,names] = load_embeddings_and_labels()
    recognizer = pickle.loads(open('models/recognizer.pickle', "rb").read())
    # vs = cv2.VideoCapture(0)
    vs = cv2.VideoCapture("http://192.168.1.2:8080//video")
    print("[INFO] starting video stream...")
    i=0
    while len(predictions) <= 100:
        try:
            (ret,frame) = vs.read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:  
                face = frame[y-5:y+h+5,x-5:x+w+5]
                resized_face = cv2.resize(face,(160,160))
                face_pixel = normalize_pixels(imagearrays=resized_face)
                sample = np.expand_dims(face_pixel,axis=0)
                embedding = facenet_model.predict(sample)
                embedding = embedding.reshape(1,-1)   
                COLORS = np.random.randint(0, 255, size=(len(label.classes_), 3), dtype="uint8")
                # perform classification to recognize the face
                preds = recognizer.predict_proba(embedding)[0]
                p = np.argmax(preds)
                proba = preds[p]
                name = label.classes_[p]
                if proba >= 0.6:
                    color = [int(c) for c in COLORS[p]]
                    cv2.imwrite(os.path.join(root_dir,'output',name + str(i) + '.jpg'),frame)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),color,5)
                    cv2.imwrite(os.path.join(root_dir,'output',name + 'face_detection' + str(i) + '.jpg'),frame)
                    text = "{}".format(name)
                    cv2.putText(frame,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
                    i+=1
                    cv2.imwrite(os.path.join(root_dir,'output',name+ 'face_recognition' + str(i)+'.jpg'),frame)
                    predictions.append(name)
                else:
                    name = "NONE"
                    color = (255,255,0)
                    text = "{}".format(name)
                    cv2.putText(frame,text,(x,y - 5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.imshow("Capture",frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        except Exception as e:
            print(e)
    vs.release()
    cv2.destroyAllWindows()

    dt = datetime.now()
    date = str(dt).split(' ')[0]
    time = str(dt).split(' ')[1]
    csv_name = 'Attendance_Details/attendance_report' + str(date) + '.csv'
    mark_attendance_obj = Mark_Attendance(csv_filename=csv_name)
    if not os.path.exists(os.path.join(root_dir,csv_name)):
        mark_attendance_obj.write_csv_header(date='Date',staff_name='Staff_Name',time='Time',status='Status')
    final_name = mode(predictions)
    status = "Present"
    df = pd.read_csv(csv_name)
    if date in str(df['Date']) and final_name in str(df['Staff_Name']):
        print("Sorry {}.Your attendance has already been recorded".format(final_name))
    else:   
        attendance_record = [date,final_name,time,status]
        mark_attendance_obj.append_csv_rows(records=attendance_record)
        print("Hello {}.Your attendance has been recorded successfully".format(final_name))

    
else:
    print("Model file not found. Embeddings.pickle file and Recognizer.pickle file must exist within models directory.")
