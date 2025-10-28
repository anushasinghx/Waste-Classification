
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
#import skimage.io
import tensorflow 
import tqdm
import glob

from tqdm import tqdm 

#from skimage.io import imread, imshow
#from skimage.transform import resize

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, BatchNormalization, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array

get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


train_o = glob.glob('C:/Anusha/PBL 3/DATASET/DATASET1/TRAIN/O/*.jpg')
a = len(train_o)
train_r = glob.glob('C:/Anusha/PBL 3/DATASET/DATASET1/TRAIN/R/*.jpg')
b = len(train_r)


# In[13]:


print("Nos of training samples: {}".format(a+b))


# In[14]:


#DataAugmentation
train_datagen = ImageDataGenerator(rescale = 1.0 / 255.0,
                                   zoom_range = 0.4,
                                   rotation_range = 10,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   validation_split = 0.2)

valid_datagen = ImageDataGenerator(rescale = 1.0 / 255.0,
                                   validation_split = 0.2)

test_datagen  = ImageDataGenerator(rescale = 1.0 / 255.0)


# In[15]:


train_dataset  = train_datagen.flow_from_directory(directory = 'C:/Anusha/PBL 3/DATASET/DATASET1/TRAIN',
                                                   target_size = (224,224),
                                                   class_mode = 'binary',
                                                   batch_size = 128, 
                                                   subset = 'training')


# In[16]:


valid_dataset = valid_datagen.flow_from_directory(directory = 'C:/Anusha/PBL 3/DATASET/DATASET1/TEST',
                                                  target_size = (224,224),
                                                  class_mode = 'binary',
                                                  batch_size = 128, 
                                                  subset = 'validation')


# In[17]:


train_dataset.class_indices


# In[15]:





# In[18]:


# Defining Model

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

base_model = ResNet50(input_shape=(224,224,3), 
                   include_top=False,
                   weights="imagenet")


# In[19]:


# Freezing Layers 

for layer in base_model.layers:
    layer.trainable=False


# In[20]:


#ViewingSummary
base_model.summary()


# In[21]:


# Defining Layers

model=Sequential()
model.add(base_model)
model.add(Dropout(0.2))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(1024,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1024,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))


# In[22]:


import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('A')


# In[23]:


#Layers Model Summary
model.summary()


# In[24]:


# Model Compile 

OPT    = tensorflow.keras.optimizers.Adam(lr=0.001)

model.compile(loss='binary_crossentropy',
              metrics=[tensorflow.keras.metrics.AUC(name = 'auc')],
              optimizer=OPT)
print('A')


# In[25]:


# Defining Callbacks

filepath = 'C:/Anusha/PBL 3/bestweights.hdf5'

earlystopping = EarlyStopping(monitor = 'val_auc', 
                              mode = 'max' , 
                              patience = 5,
                              verbose = 1)

checkpoint    = ModelCheckpoint(filepath, 
                                monitor = 'val_auc', 
                                mode='max', 
                                save_best_only=True, 
                                verbose = 1)


callback_list = [earlystopping, checkpoint]


# In[26]:


# Model Fitting 

model_history=model.fit(train_dataset,
                        validation_data=valid_dataset,
                        epochs = 5,
                        callbacks = callback_list,
                        verbose = 1)


# In[27]:


import pickle
model_path = 'C:/Anusha/PBL 3/saved_model/model'
with open(model_path +'.pkl', 'wb') as f:
    pickle.dump(model, f)
print('A')


# In[21]:


history = 'C:/Anusha/PBL 3/saved_model/history_path'
with open(history,+'.pkl' 'wb') as f:
    pickle.dump(model_history.history, f)


# In[43]:





# In[28]:


# Model Evaluation : Summarize the model loss

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left', bbox_to_anchor=(1,1))
plt.show()


# In[29]:


# Test Data 

test_data = test_datagen.flow_from_directory(directory = 'C:/Anusha/PBL 3/DATASET/DATASET1/TEST',
                                             target_size = (224,224),
                                             class_mode = 'binary',
                                             batch_size = 128)


# In[30]:


# Evaluating Loss and AUC - Test Data 

model.evaluate(test_data)


# In[34]:


from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


# In[37]:


# Test Case:1 - ORGANIC

dic = test_data.class_indices
idc = {k:v for v,k in dic.items()}

img = load_img('C:/Anusha/PBL 3/DATASET/DATASET1/TEST/O/O_12568.jpg', target_size=(224,224))
img = img_to_array(img)
img = img / 255
plt.imshow(img)
plt.axis('off')
img = np.expand_dims(img,axis=0)
answer = model.predict(img)

if answer[0][0] > 0.5:
    print("The image belongs to Recycle waste category")
else:
    print("The image belongs to Organic waste category ")


# In[39]:


# Test Case:2 - ORGANIC

dic = test_data.class_indices
idc = {k:v for v,k in dic.items()}

img = load_img('C:/Anusha/PBL 3/DATASET/DATASET1/TEST/O/O_13022.jpg', target_size=(224,224))
img = img_to_array(img)
img = img / 255
plt.imshow(img)
plt.axis('off')
img = np.expand_dims(img,axis=0)
answer = model.predict(img)

if answer[0][0] > 0.5:
    print("The image belongs to Recycle waste category")
else:
    print("The image belongs to Organic waste category ")


# In[41]:


# Test Case:3 - RECYCLABLE

dic = test_data.class_indices
idc = {k:v for v,k in dic.items()}

img = load_img('C:/Anusha/PBL 3/DATASET/DATASET1/TEST/R/R_10812.jpg', target_size=(224,224))
img = img_to_array(img)
img = img / 255
plt.imshow(img)
plt.axis('off')
img = np.expand_dims(img,axis=0)
answer = model.predict(img)

if answer[0][0] > 0.5:
    print("The image belongs to Recycle waste category")
else:
    print("The image belongs to Organic waste category ")


# In[ ]:





# In[ ]:




