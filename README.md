# Reverse-Image-Search
Objective: A system that suggests items based on the provided image.  
Example: Google lens  
Dataset used: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small  
Concepts implemented: ResNet, K-Nearest Neighbors  
Programming language: Python  
Libraries used: numpy, tensorflow, sklearn, os, streamlit, pickle  
Idea behind implementation:  

1. Import necessary libraries.
2. Extract features using ResNet.
3. Initialize the NearestNeighbors model and train it on the extracted features.
4. Predict on the new data.
