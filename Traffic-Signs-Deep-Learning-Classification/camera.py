
# import the opencv library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import sys
from tqdm import tqdm
import pickle

import pandas as pd




import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D




from PIL import Image
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

import os

from keras.models import load_model


from matplotlib.image import imread

 # Label Overview
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }
 


#image_path=sys.path[0]+'/'+image_name


model_name="model.h5"
#os.chdir(path)
model = load_model('./'+model_name)

def predict_image_from_video(image_array):

    IMG_SIZE=80

    
    #image = imread(image_path)
    img_array=image_array
    # path='/content/drive/MyDrive/Test'
    # img='00159.png'
    #print(image.shape)
    #img_array = cv2.imread(image_path ,cv2.IMREAD_GRAYSCALE)  # convert to array
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
    X_use=new_array 
    X_use = np.array(X_use).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    use=X_use.copy()    
    use=use.reshape(-1,80,80,1  ) 
    print(use.shape)
    predict_x=model.predict(use)    
    Y_pred=classes_x=np.argmax  (predict_x,axis=1)    
    plot,prediction=use,Y_pred  
    s = [str(i) for i in prediction] 
    a = int("".join(s)) 
    return a
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_class=predict_image_from_video(gray_image)
    print(type(frame))
    # Display the resulting frame
    font                   = cv2.FONT_HERSHEY_PLAIN
    #=cv2.FONT
    bottomLeftCornerOfText = (400,600)
    fontScale              = 4
    fontColor              = (0,0,255)
    lineType               = 2

    cv2.putText(frame,classes[image_class], 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)

    cv2.imshow('frame', frame)
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()