import streamlit as st
import os
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from PIL import Image


#creation of web page
st.title("Reverse Image Search")

#saving the uploaded file into a folder called uploads
def save_uploaded_file(uploaded_file):
  try: 
    with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
      f.write(uploaded_file.getbuffer())
    return 1
  except:
    return 0

#loading feature list and filenames
feature_list=np.array(pickle.load(open("embeddings.pkl", "rb")))
filenames=pickle.load(open("filenames.pkl", "rb"))

#model
model=ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model=tensorflow.keras.Sequential([
  model,
  GlobalMaxPooling2D()
])

#feature extraction
def feature_extraction(img_path, model):
  img=image.load_img(img_path, target_size=(224,224))
  img_array=image.img_to_array(img)
  expanded_img_array=np.expand_dims(img_array, axis=0)
  preprocessed_img=preprocess_input(expanded_img_array)
  result=model.predict(preprocessed_img)
  normalized_result=result/norm(result)

  return normalized_result

#recommendation
def recommend(features, feature_list):
  neighbors=NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='brute')
  neighbors.fit(feature_list)
  distances, indices = neighbors.kneighbors(features)
  return indices

#upload file
uploaded_file=st.file_uploader('Choose an image')
if uploaded_file is not None:
  if save_uploaded_file(uploaded_file):
    #display image
    display_image=Image.open(uploaded_file)
    st.image(display_image)

    #features
    features=feature_extraction(os.path.join("uploads", uploaded_file.name), model)
  
    #recommendation
    indices=recommend(features, feature_list)
    
    #dispaly recommendations
    col1, col2, col3, col4, col5=st.columns(5)

    with col1:
      st.image(filenames[indices[0][0]][7::])
    with col2:
      st.image(filenames[indices[0][1]][7::])
    with col3:
      st.image(filenames[indices[0][2]][7::])
    with col4:
      st.image(filenames[indices[0][3]][7::])
    with col5:
      st.image(filenames[indices[0][4]][7::])
      
  else:
    st.header("Some error occured in file upload")
