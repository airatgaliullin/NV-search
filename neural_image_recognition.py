import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import os



class CNN:
	def __init__(self):
		keras.backend.set_image_dim_ordering('th')
		keras.layers.core.Dropout(0.5)
		self.classifier = Sequential()
		self.classifier.add(Convolution2D(32, 3, 3, subsample=(2,2), input_shape = (3,480,480), activation = 'relu'))
		self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
		self.classifier.add(Convolution2D(64, 3, 3, subsample=(2,2),activation = 'relu'))
		self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
		self.classifier.add(Flatten())
		self.classifier.add(Dense(64,activation = 'relu'))
		self.classifier.add(Dense(2,  activation = 'softmax'))
		self.classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])


	def train(self,folder_train,folder_validate=None, saved_file='trained_explorer'):
		
		train_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip = True,vertical_flip=True,
			zoom_range=0.1, width_shift_range=0.1,height_shift_range=0.1)


		training_set = train_datagen.flow_from_directory(folder_train,
		target_size = (480, 480),
		classes=['NV','not NV'],
		batch_size = 2,
		class_mode = 'categorical')

		labels_training=training_set.classes
		labels_training=to_categorical(labels_training,2)





		if folder_validate!=None:
			validate_datagen = ImageDataGenerator(rescale = 1./255)
			
			validate_set = validate_datagen.flow_from_directory(folder_validate,
			target_size = (480, 480),
			classes=['NV', 'not NV'],
			batch_size =2,
			class_mode = 'categorical')


			labels_validate=validate_set.classes
			labels_validate==to_categorical(labels_validate,2)


			self.classifier.fit_generator(training_set,
			samples_per_epoch = len(training_set.filenames),
			nb_epoch= 25,
			validation_data= validate_set,
			nb_val_samples =len(validate_set.filenames))

			

		else:
			self.classifier.fit_generator(training_set,
			samples_per_epoch = len(training_set.filenames),
			nb_epoch= 1)

		self.classifier.save_weights(os.path.join(r'D:\measuring\analysis\scripts\Fabrication\cnn_weights',saved_file+'.h5'))


	def load_trained_model(self,load_file='trained_explorer'):
		self.classifier.load_weights(os.path.join(r'D:\measuring\analysis\scripts\Fabrication\cnn_weights',load_file+'.h5'))


	def analyze_images_from_generator(self,folder_test):
		# test_image = image.load_img(image_name, target_size = (480, 480))
		# test_image = image.img_to_array(test_image)
		# test_image = np.expand_dims(test_image, axis = 0)
		# result = self.classifier.predict_classes(test_image)


		test_datagen=ImageDataGenerator(rescale = 1./255)
		test_set=test_datagen.flow_from_directory(folder_test,
			target_size = (480, 480),
			batch_size =2,
			class_mode = 'binary')

		predictions=self.classifier.predict_generator(test_set,val_samples=len(test_set.filenames))

	def analyze_images_from_folder(self,folder_test):
		A=0
		for filename in os.listdir(folder_test):
			image_path=os.path.join(folder_test,filename)
			test_image = image.load_img(image_path, target_size = (480, 480))
			test_image = image.img_to_array(test_image)
			test_image = np.expand_dims(test_image, axis = 0)
			result = self.classifier.predict_classes(test_image)
			if result[0][0]==0:
				print filename
				A=A+1

		print 'number of blobs', A














if __name__ == '__main__':
	folder_train=r'D:\measuring\data\20180210\data\train'
	folder_validate=r'D:\measuring\data\20180210\data\validate'
	I=CNN()
	I.train(folder_train,saved_file='trained_explorer_68',
        folder_validate=folder_validate)









# classifier = Sequential()
# classifier.add(Convolution2D(32, 3, 3, subsample=(2,2), input_shape = (3,480,480), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))


# classifier.add(Convolution2D(16, 3, 3, subsample=(2,2),activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))

# # classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
# # classifier.add(MaxPooling2D(pool_size = (2, 2)))




# classifier.add(Flatten())
# classifier.add(Dense(32,activation = 'relu'))

# # classifier.add(Dropout(0.5))

# classifier.add(Dense(1,  activation = 'sigmoid'))
# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# train_datagen = ImageDataGenerator(rescale = 1./255,
# horizontal_flip = True)


# test_datagen = ImageDataGenerator(rescale = 1./255)



# training_set = train_datagen.flow_from_directory(r'D:\measuring\data\20180210\data\train',
# target_size = (480, 480),
# classes=['NV','not NV'],
# batch_size = 2,
# class_mode = 'binary')


# test_set = test_datagen.flow_from_directory(r'D:\measuring\data\20180210\data\validate',
# target_size = (480, 480),
# classes=['NV', 'not NV'],
# batch_size =1,
# class_mode = 'binary')

# classifier.fit_generator(training_set,
# samples_per_epoch = 64,
# nb_epoch= 10)
# # validation_data= test_set,
# # nb_val_samples =6)
# classifier.save_weights('first_try.h5')




# classifier.load_weights('first_try.h5')
# import numpy as np
# from keras.preprocessing import image
# test_image = image.load_img(r'D:\measuring\data\20180208\094310_scan2d\z=-3; x_c=-9; y_c=-5; search_range=2um.png', target_size = (480, 480))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = classifier.predict_classes(test_image)
# training_set.class_indices
# # if result[0][0] == 1:
# # 	prediction = 'dog'
# # else:	
# # 	prediction = 'cat'

# print result


#keras.backend.image_dim_ordering()
#set_image_dim_ordering('th')