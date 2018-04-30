import keras
import numpy as np
import os
import time
from vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras.models import load_model

my_model=load_model('my_model_animal_inception.h5')

import numpy as np
img = image.load_img("14.jpg",target_size=(299,299,3))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)
x = np.array(x)

x.shape

CLASS_INDEX = {0:"cats",1:"dogs",2:"horses",3:"Humans"}
def decode_predictions(preds):
    global CLASS_INDEX
    assert len(preds.shape) == 2 and preds.shape[1] == 4
   
    indices = np.argmax(preds, axis=-1)
    results = []
    for i in indices:
        results.append(CLASS_INDEX[i])
    return results

preds = my_model.predict(x)
print('Predicted:', decode_predictions(preds))

len(preds.shape)

from keras.preprocessing import image as image_utils
#from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
#from vgg16 import VGG16
import argparse
import cv2
import numpy as np
import os
import random
import sys

import threading

label = ""
frame = None


    
import cv2, pafy

url = "https://www.youtube.com/watch?v=4OLJe8iMLg4"
videoPafy = pafy.new(url)
best = videoPafy.getbest(preftype="webm")

cap = cv2.VideoCapture(best.url)
if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

#keras_thread = MyThread()
#keras_thread.start()

while (True):
    ret, original = cap.read()
    image = cv2.resize(original, (299, 299))
    #image = image_utils.load_img(frame, target_size=(224, 224))
    image = image_utils.img_to_array(image)

# Convert (3, 224, 224) to (1, 3, 224, 224)
# Here "1" is the number of images passed to network
# We need it for passing batch containing serveral images in real project
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)


# Classify the image
    print("[INFO] classifying image...")
    preds = my_model.predict(image)
    label = decode_predictions(preds)[0]
    print(label)

    # Display the predictions
    # print("ImageNet ID: {}, Label: {}".format(inID, label))
    cv2.putText(original, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", original)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()
