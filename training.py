"""
Train ML Model to Classify / Identify the person using extracted face embeddings
"""
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import pickle
import numpy as np
import os
from sklearn.calibration import CalibratedClassifierCV

rootdir = os.getcwd()

embeddings_path = os.path.join(rootdir,'models/embeddings.pickle')

def load_embeddings_and_labels():
    data = pickle.loads(open(embeddings_path, "rb").read())
    # encoding labels by names
    label = LabelEncoder()
    names = np.array(data["names"])                       
    labels = label.fit_transform(names)
    # getting names
    # getting embeddings
    Embeddings = np.array(data["embeddings"])
    return [label,labels,Embeddings,names]

def create_svm_model(labels,embeddings):
    model_svc = LinearSVC()
    recognizer = CalibratedClassifierCV(model_svc)   
    recognizer.fit(embeddings,labels)
    return recognizer


[label,labels,Embeddings,names] = load_embeddings_and_labels()
recognizer = create_svm_model(labels=labels,embeddings=Embeddings)
f1 = open('models/recognizer.pickle', "wb")
f1.write(pickle.dumps(recognizer))
f1.close()
print("Training done successfully")



