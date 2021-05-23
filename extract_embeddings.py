import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
import pickle


rootdir = os.getcwd()
dataset_dir = os.path.join(rootdir,'dataset')

model_path = os.path.join(rootdir,'models/facenet_keras.h5')
facenet_model = load_model(model_path)

categories = os.listdir(dataset_dir)

def check_pretrained_file(embeddings_model):
	data = pickle.loads(open(embeddings_model,"rb").read())
	names = np.array(data["names"])
	unique_names = np.unique(names).tolist()
	return [data,unique_names]

def get_remaining_names(unique_names):
	remaining_names = np.setdiff1d(categories,unique_names).tolist()
	return remaining_names

def get_all_face_pixels():
	image_ids = []
	image_paths = []
	image_arrays = []
	names = []
	for category in categories:
		path = os.path.join(dataset_dir,category)
		for img in os.listdir(path):
			img_array = cv2.imread(os.path.join(path,img))
			image_paths.append(os.path.join(path,img))
			image_ids.append(img)
			image_arrays.append(img_array)
			names.append(category)
	return [image_ids,image_paths,image_arrays,names]


def get_remaining_face_pixels(remaining_names):
	image_ids = []
	image_paths = []
	image_arrays = []
	names = []
	face_ids = []
	if len(remaining_names) != 0:	
		for category in remaining_names:
			path = os.path.join(dataset_dir,category)
			for img in os.listdir(path):
				img_array = cv2.imread(os.path.join(path,img))
				image_paths.append(os.path.join(path,img))
				image_ids.append(img)
				image_arrays.append(img_array)
				names.append(category)
		return [image_ids,image_paths,image_arrays,names]
	else:
		return None


def normalize_pixels(imagearrays):
	face_pixels = np.array(imagearrays)
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	return face_pixels



embeddings_model_file = os.path.join(rootdir,"models/embeddings.pickle")
if not os.path.exists(embeddings_model_file):
	[image_ids,image_paths,image_arrays,names] = get_all_face_pixels()
	face_pixels = normalize_pixels(imagearrays = image_arrays)   
	embeddings = []
	for (i,face_pixel) in enumerate(face_pixels):
		sample = np.expand_dims(face_pixel,axis=0)
		embedding = facenet_model.predict(sample)
		new_embedding = embedding.reshape(-1)
		embeddings.append(new_embedding)
		data = {"paths":image_paths, "names":names,"imageIDs":image_ids,"embeddings":embeddings}
	f = open('models/embeddings.pickle' , "wb")
	f.write(pickle.dumps(data))
	f.close()

else:
	[old_data,unique_names] = check_pretrained_file(embeddings_model_file)
	remaining_names = get_remaining_names(unique_names)
	data = get_remaining_face_pixels(remaining_names)
	if data != None:
		[image_ids,image_paths,image_arrays,names] = data
		face_pixels = normalize_pixels(imagearrays = image_arrays)
		embeddings = []
		for (i,face_pixel) in enumerate(face_pixels):
			sample = np.expand_dims(face_pixel,axis=0)
			embedding = facenet_model.predict(sample)
			new_embedding = embedding.reshape(-1)
			embeddings.append(new_embedding)
		new_data = {"paths":image_paths, "names":names,"imageIDs":image_ids,"embeddings":embeddings}
		combined_data = {"paths":[],"names":[],"face_ids":[],"imageIDs":[],"embeddings":[]}
		combined_data["paths"] = old_data["paths"] + new_data["paths"]
		combined_data["names"] = old_data["names"] + new_data["names"]
		combined_data["face_ids"] = old_data["face_ids"] + new_data["face_ids"]
		combined_data["imageIDs"] = old_data["imageIDs"] + new_data["imageIDs"]
		combined_data["embeddings"] = old_data["embeddings"] + new_data["embeddings"]

		f = open('models/embeddings.pickle' , "wb")
		f.write(pickle.dumps(combined_data))
		f.close()
	else:
		print("No new data found... Embeddings has already extracted for this user")
		

