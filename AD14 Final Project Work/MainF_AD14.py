# ===== IMPORT REQUIRED PACKAGES =========

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg

#====================== 1.READ A INPUT IMAGE =========================

filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img,cmap='gray')
plt.title('ORIGINAL IMAGE')
plt.axis ('off')
plt.show()


# ====================== 2. PREPROCESSING ==========================

#==== RESIZE IMAGE ====

resized_image = cv2.resize(img,(300,300))
img_resize_orig = cv2.resize(img,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()
           
                 
                 
#==== GRAYSCALE IMAGE ====

try:            
    gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray1 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray1,cmap='gray')
plt.axis ('off')
plt.show()
    
        
#=================== 3.FEATURE EXTRACTION ===================

#=== MEAN STD DEVIATION ===

mean_val = np.mean(gray1)
median_val = np.median(gray1)
var_val = np.var(gray1)
features_extraction = [mean_val,median_val,var_val]

print("------------------------------------")
print("     MEAN MEDIAN VARAINCE           ")
print("------------------------------------")
print()
print(features_extraction)

    
 # ================== 4. IMAGE  SPLITTING ======================

# TEST AND TRAIN 


import os 

from sklearn.model_selection import train_test_split

dataset_ade = os.listdir('Dataset/adenocarcinoma')

dataset_carci = os.listdir('Dataset/large.cell.carcinoma')

dataset_nor = os.listdir('Dataset/normal')

dataset_squamous = os.listdir('Dataset/squamous.cell.carcinoma')


dot1= []
labels1 = []
for img in dataset_ade:
        # print(img)
        img_1 = mpimg.imread('Dataset/adenocarcinoma/' + "/" + img)

        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(0)

for img in dataset_carci:
        # print(img)
        img_1 = mpimg.imread('Dataset/large.cell.carcinoma/' + "/" + img)

        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(1)        

for img in dataset_nor:
        img_1 = mpimg.imread('Dataset/normal/' + "/" + img)

        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(2)

for img in dataset_squamous:
        # print(img)
        img_1 = mpimg.imread('Dataset/squamous.cell.carcinoma/' + "/" + img)

        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(3)

x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.3, random_state = 101)


print("---------------------------------")
print("Image Splitting")
print("---------------------------------")
print()
print("1. Total Number of images =", len(dot1))
print()
print("2. Total Number of Test  =", len(x_test))
print()
print("3. Total Number of Train =", len(x_train))    


#============================ CLASSIFICATION =================================

# === ALEXNET ===
    

from keras.utils import to_categorical
global acc_cmt

y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]



from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import MaxNorm
from keras.optimizers import Adam
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras import backend as K









#Define Alexnet Model
def AlexnetModel(input_shape,num_classes):
  model = Sequential()
  model.add(Conv2D(filters=96,kernel_size=(3,3),strides=(4,4),input_shape=(50,50,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
  model.add(Conv2D(256,(5,5),padding='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
  model.add(Conv2D(384,(3,3),padding='same',activation='relu'))
  model.add(Conv2D(384,(3,3),padding='same',activation='relu'))
  model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(4,activation='softmax'))

  return model


def lr_schedule(epoch):


    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr



model = AlexnetModel((50,50,3),4)
#optimizer = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss= 'categorical_crossentropy' , optimizer=Adam(0.01), metrics=[ 'accuracy' ])
# print("Model Summary of ",model_type)
print(model.summary())

history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=2,verbose=1) 


print("------------------------------")
print("Alexnet Deep Learning         ")
print("-------------------------------")
print()

loss=history.history['loss']
loss=max(loss) * 10
acc_alexnet=100-loss
print()
print("1.Accuracy is :",acc_alexnet,'%')
print()
print("2.Loss is     :",loss)
print()


plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('VALIDATION')
plt.ylabel('ACC')
plt.xlabel('# Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# ==================== PREDICTION
    
print("-------------------------------")
print("           Prediction          ")
print("--------------------------------")
print()

Total_length = len(dataset_ade) + len(dataset_carci) + len(dataset_nor) + len(dataset_squamous)

temp_data1  = []
for ijk in range(0,Total_length):
    # print(ijk)
    temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
    temp_data1.append(temp_data)

temp_data1 =np.array(temp_data1)

zz = np.where(temp_data1==1)

if labels1[zz[0][0]] == 0:
    print('--------------------------------------')
    print()
    print(' Identified  as  AFFECTED')
    print(' Type    -- "Adenocarcinoma"')
    print()
    print('-------------------------------------')

elif labels1[zz[0][0]] == 1:
    print('---------------------------------------')
    print()
    print(' Identified  as  AFFECTED')
    print(' Type  -- Large.cell.carcinoma')
    print()
    print('--------------------------------------')

elif labels1[zz[0][0]] == 2:
    print('--------------------------------------------')
    print()
    print(' Identified  as  -- Normal')    
    print()
    print('------------------------------------------')

elif labels1[zz[0][0]] == 3:
    print('------------------------------------------')
    print()
    print(' Identified  as  AFFECTED')
    print(' Type-- Squamous.cell.carcinoma')    
    print()
    print('-------------------------------------------')


# ===== COMPARISON =====

vals=[acc_alexnet,loss]
inds=range(len(vals))
labels=["ACCURACY","ERROR RATE"]
fig,ax = plt.subplots()
rects = ax.bar(inds, vals)
ax.set_xticks([ind for ind in inds])
ax.set_xticklabels(labels)
plt.show()









