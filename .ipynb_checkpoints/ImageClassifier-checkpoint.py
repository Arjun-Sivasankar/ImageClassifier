#!/usr/bin/env python
# coding: utf-8

# ## INSTALL DEPENDENCIES

# In[1]:


get_ipython().system('pip install tensorflow tensforflow-gpu opencv-python matplotlib')


# In[2]:


import os
import tensorflow as tf


# In[3]:


# os.path.join('data', 'happy')


# In[4]:


# GPU's that i have:
gpus = tf.config.experimental.list_physical_devices('GPU')
gpus


# In[5]:


## CPU's that i have:
cpus = tf.config.experimental.list_physical_devices('CPU')
cpus


# In[6]:


## Avoid OOM errors by setting the GPU Memory Consumption Growth :
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# ## Remove dodgy images

# In[7]:


import cv2
import imghdr ## checks file extension
import matplotlib.pyplot as plt


# In[8]:


data_dir = 'data'


# In[9]:


image_ext = ['jpeg', 'jpg', 'bmp', 'png']


# In[10]:


os.listdir(data_dir)


# In[11]:


os.listdir(os.path.join(data_dir, 'happy'))


# In[12]:


os.listdir(os.path.join(data_dir, 'sad'))


# In[13]:


## No. of happy pics:
len(os.listdir(os.path.join(data_dir, 'happy')))


# In[14]:


## No. of sad pics:
len(os.listdir(os.path.join(data_dir, 'sad')))


# #### Reading an image:
#     

# In[15]:


img = cv2.imread(os.path.join('data', 'happy', 'pexels-photo-4611670.jpeg'))
img


# In[16]:


type(img)


# In[17]:


img.shape


# This means that the image height = 6240px and width = 4160px

# In[18]:


plt.imshow(img)


# OpenCV reads an image as 'BGR' and matplotlib expects it to be in "RGB". That's why this looks a biut bizarre.

# In[19]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# In[20]:


for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        # print(image_path)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_ext:
                print("Image not in the extension list {}".format(image_path))
                os.remove(image_path)
        except exception as e:
            print("Issue with image {}".format(image_path))                      


# In[21]:


len(os.listdir(os.path.join(data_dir, 'happy')))


# Therefore, 6 pics from 'data\happy' were removed.

# In[22]:


len(os.listdir(os.path.join(data_dir, 'sad')))


# Therefore, 8 pics from 'data\sad' were removed.

# ## LOAD DATASET:

# In[23]:


get_ipython().run_line_magic('pinfo2', 'tf.data.Dataset')


# In[24]:


import numpy as np


# In[25]:


get_ipython().run_line_magic('pinfo2', 'tf.keras.utils.image_dataset_from_directory')


# In[26]:


data = tf.keras.utils.image_dataset_from_directory('data')


# In[27]:


data_iterator = data.as_numpy_iterator()
data_iterator


# In[28]:


batch = data_iterator.next()
batch


# In[29]:


len(batch)


# Notice, the length of the batch is 2. That is because - one belongs to the image representation as numpy arrays and the other belongs to the labels.

# Batch of images: 

# In[30]:


# Images represented as numpy arrays
batch[0].shape    ## shape of the batch of image representation


# In[31]:


batch[1]          ## shape of the labels

## Class 0 ==> HAPPY people
## CLass 1 ==> SAD people


# In[32]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# Note: the labels are written at the top of the images.

# ## PREPROCESS DATA:

# ### * Scale Data:

# In[33]:


data = data.map(lambda x, y: (x/255, y))
data


# In[34]:


data.as_numpy_iterator().next()


# In[35]:


len(data)


# ### * Split Data:

# In[65]:


train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)+1


# In[66]:


train_size+val_size+test_size


# In[67]:


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


# ## BUILD DEEP LEARNING MODEL:

# In[39]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# In[40]:


model = Sequential()


# In[41]:


model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[42]:


model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])   ## using the "Adam" optimiser


# In[43]:


model.summary()


# #### Train

# In[44]:


logdir='logs'


# In[45]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[46]:


hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])


# In[47]:


hist.history


# #### Plot Performance:

# In[48]:


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[49]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# ## Evaluate Performance:

# #### 1. Evaluate

# In[50]:


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# In[69]:


precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()


# In[70]:


len(test)


# In[71]:


for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)


# In[72]:


print(f'Precision:{precision.result().numpy()}, Recall: {recall.result().numpy()}, Accuracy: {accuracy.result().numpy()}')


# #### 2. Test:

# Test ===> Happy

# In[73]:


import cv2


# In[76]:


img = cv2.imread('happy_test.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# In[85]:


resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[89]:


resize 


# Note the shape

# In[90]:


resize.shape


# In[91]:


np.expand_dims(resize, 0)


# Now, note the shape

# In[93]:


np.expand_dims(resize, 0).shape


# In[86]:


yhat = model.predict(np.expand_dims(resize/255, 0))


# In[87]:


yhat


# In[88]:


if yhat > 0.5: 
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')


# Test ===> Sad (No need to repeat this (can implement in the above snippets)...But anyways!)

# In[95]:


img = cv2.imread('sad_test.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# In[96]:


resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[97]:


np.expand_dims(resize, 0)


# In[98]:


yhat = model.predict(np.expand_dims(resize/255, 0))


# In[99]:


yhat


# In[100]:


if yhat > 0.5: 
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')


# ## SAVE THE MODEL:
# 

# In[101]:


from tensorflow.keras.models import load_model


# In[102]:


model.save(os.path.join('models','imageclassifier.h5'))


# In[104]:


new_model = load_model('models\imageclassifier.h5')


# In[106]:


yhat_new = new_model.predict(np.expand_dims(resize/255, 0))
yhat_new


# In[107]:


if yhat_new > 0.5: 
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')


# In[ ]:




