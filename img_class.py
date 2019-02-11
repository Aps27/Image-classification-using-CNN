# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(keras.layers.Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

'''
img X feature detector = feature map
32 feature detectors with 3X3 rows, cols of feature detector
input_shape is dimensions of input pictures (convert all images to fixed size) 3D arrays : RGB
input_shape has 3 channels followed by dimensions coz of tensorflow backend, opposite for theano backend
'''

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

'''
2 X 2 we keep info and precise 

'''

# Adding a second convolutional layer
classifier.add(keras.layers.Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(128, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
'''
128 : power of 2 (inout)
1 : dog or cat (output)
'''
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

''' Part 2 - Fitting the CNN to the images '''

# Keras has an image preprocessing library to prevent overfitting
# code from keras.io
# uses data augmentation

from keras.preprocessing.image import ImageDataGenerator

# image augmentation section COMPULSARY
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         epochs = 5, #25
                         validation_data = test_set,
                         validation_steps = 2000)


classifier.save('model.h5') # in case we need same model for further processing without compilation

import cv2 # openCV
import numpy as np
#model = load_model('model.h5')
img = cv2.imread('test.jpg')
img = cv2.resize(img,(64,64))
img = np.reshape(img,[1,64,64,3])

classes = classifier.predict_classes(img)

'''
 In this model, as we are using a binary cross entropy function, we would get the output as 1 or 0 - 2 classes that corresponds 
 to one of the images. For multiple classes, we use categorical cross entropy function.
'''
print(classes)
if classes == 1:
    print("Picture is of a mountain")
else:
    print("Picture is of a river")

