import numpy as np
import cv2

#data pre-processing

train_f = []
train_l=[]
test_f=[]
test_l=[]

#inserting lables of cardboard data
for j in range(1,400):
    train_l.append(0)
#inserting lable of glass data    
for j in range(1,400):
    train_l.append(1)     
#inserting lable of metal data    
for j in range(1,400):
    train_l.append(2)      
#inserting lable of paper data    
for j in range(1,400):
    train_l.append(3)     
#inserting lable of plastic data    
for j in range(1,400):
    train_l.append(4)         

train_l=np.array(train_l)          #converting list to array

################################# getting training feature
#inserting features of cardboard data for training
for i in range(1,400):
    path='trash classification/cardboard/cardboard%d.jpg'%(i)    
    image = cv2.imread(path,0)    
    train_f.append(image)

#inserting features of glass data for training
for i in range(1,400):
    path ='/trash classification/glass/glass%d.jpg'%(i)
    image = cv2.imread(path,0)   
    train_f.append(image) 

#inserting features of metal data for training
for i in range(1,400):
    path ='trash classification/metal/metal%d.jpg'%(i)
    image = cv2.imread(path,0)   
    train_f.append(image) 

#inserting features of paper data for training
for i in range(1,400):
    path ='trash classification/paper/paper%d.jpg'%(i)
    image = cv2.imread(path,0)   
    train_f.append(image) 

#inserting features of plastic data for training
for i in range(1,400):
    path ='trash classification/plastic/plastic%d.jpg'%(i)
    image = cv2.imread(path,0)   
    train_f.append(image) 

train_f = np.array(train_f)           #converting list to array


############################ getting testing feature 
#inserting features of cardboard data for testing
for i in range(400,404):
    path='trash classification/cardboard/cardboard%d.jpg'%(i)    
    image = cv2.imread(path,0)    
    test_f.append(image)

#inserting features of glass data for testing
for i in range(400,502):
    path ='trash classification/glass/glass%d.jpg'%(i)
    image = cv2.imread(path,0)   
    test_f.append(image) 

#inserting features of metal data for testing
for i in range(400,411):
    path ='trash classification/metal/metal%d.jpg'%(i)
    image = cv2.imread(path,0)   
    test_f.append(image) 

#inserting features of paper data for testing
for i in range(400,595):
    path ='trash classification/paper/paper%d.jpg'%(i)
    image = cv2.imread(path,0)   
    test_f.append(image) 

#inserting features of plastic data for testing
for i in range(400,483):
    path ='trash classification/plastic/plastic%d.jpg'%(i)
    image = cv2.imread(path,0)   
    test_f.append(image) 
   
test_f = np.array(test_f)                   #converting list to array 

###############################getting testing lables
#inserting lables of cardboard data
for j in range(400,404):
    test_l.append(0)
#inserting lable of glass data    
for j in range(400,502):
    test_l.append(1)     
#inserting lable of metal data    
for j in range(400,411):
    test_l.append(2)      
#inserting lable of paper data    
for j in range(400,595):
    test_l.append(3)     
#inserting lable of plastic data    
for j in range(400,483):
    test_l.append(4)         

test_l=np.array(test_l)          #converting list to array


#############   ANN impementation
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten


#normalizing our data
train_f = keras.utils.normalize(train_f)
test_f = keras.utils.normalize(test_f)

classifier = Sequential()
classifier.add(Flatten())

#adding input layer and hidden layer
classifier.add(Dense(units = 500,activation='relu',kernel_initializer = 'uniform',input_dim =384))
#adding second hidden layer
classifier.add(Dense(units = 500,activation='relu',kernel_initializer = 'uniform'))
#adding third hidden layer
classifier.add(Dense(units = 500,activation='relu',kernel_initializer = 'uniform'))
#adding output layer
classifier.add(Dense(units = 5,activation='softmax',))
#compiling ANN
classifier.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])

#training of ANN
classifier.fit(train_f,train_l,epochs = 5)

#evaluating ANN on testing data
val_loss,val_acc = classifier.evaluate(test_f,test_l)
print(val_loss,val_acc) 





