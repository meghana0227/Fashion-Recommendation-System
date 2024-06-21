import tensorflow
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import pickle


#the weights that are obtained during the training of the model with imagenet dataset are used
#include_top=False indicates that the last layer of the model is removed
#input shape is resized to the provided shape

model=ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

#not training the model
model.trainable=False

model=tensorflow.keras.Sequential([
  model,
  GlobalMaxPooling2D()
])

#resnet is basically used for extracting the features

def extract_features(img_path, model):
  #reading or parsing the image
  img=image.load_img(img_path, target_size=(224,224))

  #converting the image to array
  img_array=image.img_to_array(img)

  #resnet takes batch of images as input. so, in the next step we are creating a batch consisting of single image
  expanded_img_array=np.expand_dims(img_array, axis=0)

  #preprocess_input converts the image features to imagenet dataset features
  preprocessed_img=preprocess_input(expanded_img_array)

  #prediction
  result=model.predict(preprocessed_img).flatten()

  #normalization -> bringing into same range
  normalized_result=result/norm(result)

  return normalized_result

#getting file names
filenames=[]

for file in os.listdir('images'):
  filenames.append(os.path.join('images', file))

#creating feature list which consists of features of each image
feature_list=[]

for file in filenames:
  feature_list.append(extract_features(file, model))


#saving filenames and feature_list in separate files
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))





