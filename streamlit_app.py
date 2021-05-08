import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import glob


import math

import random
import matplotlib.image as mpimg
from PIL import Image
from keras.models import model_from_json

def predict(path):

    X_test = []
    X_test_id = []
    d_class=["normal driving","texting - right" , "talking on the phone - right" ,"texting - left", "talking on the phone - left", "operating the radio", "drinking", "reaching behind", "hair and makeup", "talking to passenger"]

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (10,0,200,230)
    # img=mpimg.imread(image)
    #fl = os.path.basename(path)
    # img_array = np.array(image)
    import socket   
    hostname = socket.gethostname()   
    IPAddr = socket.gethostbyname(hostname)
    img = cv2.imread(path)
    # Reduce size
    img1 = cv2.resize(img, (100, 100), cv2.INTER_LINEAR)
    st.write("Initially, the uploaded image is resized to a particular resolution i.e. 100x100.")
    st.image(img1)
    st.text("(100x100)")

    # GrabCut and Resize
    X_test.append(img1)
    mask = np.zeros(img1.shape[:2],np.uint8)
    cv2.grabCut(img1,mask,rect,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img1 = img1*mask2[:,:,np.newaxis]
    # img = cv2.resize(img, (100, 100), cv2.INTER_LINEAR)
    st.write("Then the resized image is went through image segmentation using GrabCut.")
    st.image(img1)
    st.text("(100x100 with background noise eliminated.)")

    test_data = np.array(X_test, dtype=np.uint8)
    test_data = test_data.astype('float16')   
   
    # model=train_model_5()
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    weights_path = os.path.join("weights_5.h5")
    model.load_weights(weights_path)

    test_prediction = model.predict(test_data,verbose=1)
    result=str(np.argmax(test_prediction))
    return img,d_class[int(result)]



st.title('Distracted Driver Detection')
st.sidebar.title("Welcome to:")
st.sidebar.write("Distracted Driver Detection Using Image Segmentation and Transfer Learning.")
st.sidebar.header("Developed By:")
st.sidebar.text("Aditya Prasad Tripathy-(1701106508)\nDebabrata Tripathy-(1701106447)\nSoumyajit Bal-(1701106368)\nSidhartha Bibekananda Dash-(1701106295)")
st.sidebar.header("\nIn the guidance of:")
st.sidebar.text("Dr. Sanjit Kumar Dash")

if st.button("Explore the data:"):
    link="<span >The dataset is provided by <a style='text-decoration: none;' href='https://www.kaggle.com/c/state-farm-distracted-driver-detection/submissions?sortBy=date&group=successful&page=1'>Kaggle State Farm.</a></span>"
    st.markdown(link, unsafe_allow_html=True)
    img_list = pd.read_csv('driver_imgs_list.csv')
    img_list['class_type'] = img_list['classname'].str.extract('(\\d)',expand=False).astype(float)
    img_list.hist('class_type',alpha=0.5,layout=(1,1),bins=9)
    img_list

image=st.file_uploader("Pick a  image for prediction:", type=["png","jpg","jpeg"])
if image is not None:
  st.header("Data Pre-processing:")
  st.write("The original uploaded image is:")
  st.image(image)
  with open('img','wb') as f:
    f.write(image.read())
  path2 = os.path.join('./img')
  img,result=predict(path2)
  plt.title('the prediction is '+result)
  plt.imshow(img)
  plt.savefig('resized.png')
  st.header("Data Prediction:")
  st.image("./resized.png")
#for removing the hamburger menu and the footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div.viewerBadge_link__1S137{visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
