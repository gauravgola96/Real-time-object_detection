
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

model = VGG16(include_top=False, weights='imagenet')

model.summary()

#loading files from path

path = "C:\\Users\\User.1\\Documents\\Gaurav\\DL\\Transfer learning\\Transfer-Learning-in-keras---custom-data-master\\data"

data_dir = os.listdir(path)

data_dir


img_list1 = []
for data in data_dir:
    img_list = os.listdir(path+"\\"+data)
    for img in img_list:
        img_path = path + "\\" + data+"\\" + img
        img = image.load_img(img_path,target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        x = preprocess_input(x)
        img_list1.append(x)
        
        

img_data = np.array(img_list1)

img_data.shape

img_data = np.rollaxis(img_data,axis=1,start=0)

img_data.shape

# 1 is not required
img_data = img_data[0]

img_data.shape


# Define the number of classes
num_classes = 4
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:202]=0
labels[202:404]=1
labels[404:606]=2
labels[606:]=3

names = ['cats','dogs','horses','humans']

# convert class labels to on-hot encoding
#categforical data
Y = np_utils.to_categorical(labels, num_classes)

x,y = shuffle(img_data,Y,random_state=2)

x = x[:200]

y = y[:200]



print(x.shape)
print(y.shape)



X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#own additional layers

image_input = Input(shape=(224,224,3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()

last_layer = model.get_layer(name="fc2").output

out = Dense(num_classes, activation='softmax', name='output')(last_layer)
custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()

for layers in custom_vgg_model.layers[:-1]:
    layers.trainable = False
    

custom_vgg_model.summary()

custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

h = custom_vgg_model.fit(x=X_train,y=y_train,batch_size=12,verbose=1,epochs=5,validation_data=(X_test,y_test))

y_pred = custom_vgg_model.predict_on_batch(X_test)

