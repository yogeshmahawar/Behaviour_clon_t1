
# coding: utf-8

# In[2]:


#Read the data
import pandas as pd
import numpy as np
import cv2
from PIL import Image

#df= pd.read_csv('driving_log.csv')
#heading = ['center','left','right']
#image_set = []
#measurements  = []

def load_images(df,heading):
    img_paths = df[heading]    
    images = np.array([np.array(Image.open(fname)) for fname in img_paths])   
    return images

#img_paths = df['center']    
#image_set = np.array([np.array(Image.open(fname)) for fname in img_paths])
#measurements = np.array(df['steering'])
#print(img_paths)                        
                        
#aug_images, aug_measures = [] , []

#x_train = np.array(image_set)
#y_train = measurements.astype(float)



# In[7]:


import csv
from PIL import Image
samples = []
car_images = []
action_measures = []


with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


for row in samples[1:]:        
    steering_center = float(row[3])
    throttle = float(row[4])
    brake = float(row[5])
    speed = float(row[6])
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    # read in images from center, left and right cameras
    path = "" # fill in the path to your training IMG directory
    img_center = np.asarray(Image.open(row[0].strip()))
    action_center = [steering_center,throttle,brake,speed]    
    img_left = np.asarray(Image.open(row[1].strip())
    action_left = [steering_left,throttle,brake,speed]
    #print(row[0].strip())
    img_right = np.asarray(Image.open(row[2].strip())
    action_right = [steering_center,throttle,brake,speed]
    # add images and angles to data set
    car_images.extend([img_center, img_left, img_right])
    action_measures.extend([action_center,action_left,action_right])        


print(action_measures[0])        
print(car_images[0])


# In[ ]:



aug_images = []
aug_measures = []
for img,measures in zip(car_images, action_measures):
    aug_images.append(img)
    aug_measures.append(measures)    
    aug_images.append(cv2.flip(img,1))
    measures[0] = measures[0]* -1
    aug_measures.append(measures)

print(aug_images.shape)
print(aug_measures.shape)


# In[ ]:


from sklearn.utils import shuffle
def generator(aug_images,aug_measures, batch_size=16):
    num_samples = len(aug_images)
    while 1: # Loop forever so the generator never terminates
        shuffle(aug_images,aug_measures)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# In[15]:


#Create model

from keras.models import  Sequential
from keras.layers import  Dense, Flatten, Lambda


#to be defined as nvidia CUDANN
model = Sequential()
model.add(Lambda( lambda x: x/ 255.0-0.5 , input_shape = (160,320,3)))

model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse',optimizer = 'adam')
model.summary()
model.fit(x_train,y_train, validation_split = 0.2 , shuffle='true')

model.save('model.h5')

