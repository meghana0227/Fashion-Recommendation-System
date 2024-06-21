import pickle
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

#loading filenames and feature list
feature_list=np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames=pickle.load(open('filenames.pkl', 'rb'))

#model
model=ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable=False
model=tensorflow.keras.Sequential([
  model,
  GlobalMaxPooling2D()
])

#test image preprocessing
img=image.load_img('sample/jacket.jpg', target_size=(224,224))
img_array=image.img_to_array(img)
expanded_img=np.expand_dims(img_array, axis=0)
preprocessed_img=preprocess_input(expanded_img)
result=model.predict(preprocessed_img).flatten()
normalized_result=result/norm(result)

#finding the nearest neighbours
#brute force -> distance is calculated with each image
neighbors=NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])
#display of recommendations
for file in indices[0]:
  temp_img=cv2.imread(filenames[file][7::])
  cv2.imshow('output', temp_img)
  cv2.waitKey(0)