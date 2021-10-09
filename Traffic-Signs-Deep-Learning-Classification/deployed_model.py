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




def predict_image(image_path):

    IMG_SIZE=80

    
    image = imread(image_path)  
    # path='/content/drive/MyDrive/Test'
    # img='00159.png'
    #print(image.shape)
    img_array = cv2.imread(image_path ,cv2.IMREAD_GRAYSCALE)  # convert to array
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
    print("---------------------PREDICTION------------------")
    print("Predicted traffic sign is ",a)
    new_arraytemp = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_arraytemp, cmap='gray')
    plt.show()


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
    # print("---------------------PREDICTION------------------")
    # print("Predicted traffic sign is ",a)
    # new_arraytemp = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    # plt.imshow(new_arraytemp, cmap='gray')
    # plt.show()


# path="/content/drive/MyDrive/trained_model"
# image_path='/content/drive/MyDrive/Test/00159.png'

# path=sys.path[0]

print("BOTH THE IMAGE AND MODEL SHOULD BE IN THE CODE FOLDER")
print("YOU SHOULD ENTER THE FILE TYPE AS WELL, ex: image.png")
model_name=input("model name=")
image_name=input("image_name=")
image_path=sys.path[0]+'/'+image_name



#os.chdir(path)
model = load_model('./'+model_name)






predict_image(image_path)



