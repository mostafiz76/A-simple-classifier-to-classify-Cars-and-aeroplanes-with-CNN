import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import time

start = time.time()

#Define Path
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
test_path = './v_data/train/planes'

#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)

#Define image parameters
img_width, img_height = 224, 224

#Prediction Function
def predict(file):
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0][0]
    print(array[0][0])

    #answer = np.argmax(array)
    #print(answer)
    if result >=0.5:
        print("Predicted: plane")
    else:
        print("Predicted: car")
    #elif answer == 2:
    #    print("Predicted: soccer_ball")
    
  #return answer
 result = predict('./v_data/train/planes/1.jpg')
