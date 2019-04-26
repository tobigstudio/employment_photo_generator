from keras import models
from keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense
from keras.preprocessing.image import img_to_array, load_img

class checker:
	def __init__(self, model_file):
		self.width = 32
		self.height = 32
		self.channels = 3

		self.load_model(model_file)


	def load_model(self, model_file):
		self.model = models.Sequential()
		self.model.add(Conv2D(32, kernel_size=(3, 3),
		                 activation='relu',
		                 input_shape=(self.height, self.width, self.channels)))
		self.model.add(Conv2D(64, (3, 3), activation='relu'))
		self.model.add(MaxPool2D(pool_size=(2, 2)))
		self.model.add(Conv2D(64, (3, 3), activation='relu'))
		self.model.add(MaxPool2D(pool_size=(2, 2)))
		self.model.add(Flatten())
		self.model.add(Dense(128, activation='relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(1, activation='sigmoid'))
		self.model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=["accuracy"])

		self.model.load_weights(model_file)


	def is_suit(self, file):
		img = load_img(file, target_size=(self.height, self.width))
		x = img_to_array(img)  
		x = x.reshape((1, self.height, self.width, self.channels))
		x /= 255

		y = self.model.predict_classes(x)
		if y == 0 :
			return False
		elif y == 1 :
			return True


