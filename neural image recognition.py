from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
classifier = Sequential()
classifier.add(Convolution2D(64, 3, 3, input_shape = (3,64,64))) #activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim =64, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


train_datagen = ImageDataGenerator(rescale = None,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)


test_datagen = ImageDataGenerator(rescale = None)

training_set = train_datagen.flow_from_directory('train',
target_size = (64, 64),
classes=['NV','not NV'],
batch_size = 32,
class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test',
target_size = (64, 64),
classes=['NV','not NV'],
batch_size = 32,
class_mode = 'binary')

classifier.fit_generator(training_set,
samples_per_epoch = 2,
nb_epoch= 25,
validation_data= test_set,
nb_val_samples =2)


# import numpy as np
# from keras.preprocessing import image
# test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = classifier.predict(test_image)
# training_set.class_indices
# if result[0][0] == 1:
# prediction = 'dog'
# else:
# prediction = 'cat'