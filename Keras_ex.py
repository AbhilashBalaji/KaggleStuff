
from __future__ import print_function
import csv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
x_train=[]
y_train=[]
x_test=[]
with open('train.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        x_train.append(line[1:])
        y_train.append(line[0])

with open('test.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader :
        x_test.append(line)

#np.frombuffer(test) 
batch_size = 128
num_classes = 10
epochs = 1


# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
#print(x_test)
x_train = x_train.reshape(42000, 784)
x_test = x_test.reshape(28000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    )
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
values=model.predict(x_test)
y_pred=[np.argmax(value) for value in values]
i=0
with open('ans.csv','w') as csvfile:
    writer=csv.writer(csvfile)
    for i in range(len(y_pred)):
        writer.writerow([i+1,y_pred[i]])
    
