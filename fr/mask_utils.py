import os
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras.layers import *

class MaskDetector(object):
	def __init__(self, base_path, model_file='mask_detector.model', weights_file='mask_detector.weights'):
		self.base_path = base_path
		self.model_file = os.path.join(self.base_path, model_file) 
		self.weights_file = os.path.join(self.base_path, weights_file)

		try:
			print('[INFO] Loading mask detector model from model file ... ')
			self.mask_detector = load_model(self.model_file)
		except:
			print('[INFO] Loading mask detector model from weights file ... ')
			self.mask_detector = self._model_arch()
			self.mask_detector.load_weights(self.weights_file)

	def _model_arch(self):
		baseModel = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
		headModel = baseModel.output 
		headModel = AveragePooling2D(pool_size=(7,7))(headModel)
		headModel = Flatten()(headModel)
		headModel = Dense(128, activation='relu')(headModel)
		headModel = Dropout(0.5)(headModel)
		headModel = Dense(2, activation='softmax')(headModel)

		model = Model(inputs=baseModel.input, outputs=headModel)
		return model

	def _preprocess(self, image):
		'''
			image : ndarray, bgr image of the cropped face
		'''
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (224, 224))
		image = img_to_array(image)
		image = preprocess_input(image)

		return image 

	def predict(self, image):
		image = self._preprocess(image)
		label = self.mask_detector.predict(np.array([image]))
		label = np.argmax(label[0])
		if(label == 1):
			label = 'No Mask'
		else:
			label = 'Mask'

		return label