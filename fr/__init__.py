import os
import sys
import cv2
import glob
import time
import json
import shutil
import requests
import traceback
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
from tensorflow.keras.applications import MobileNetV2 

from imutils.video import WebcamVideoStream
from scipy.spatial.distance import cdist

from .clf import EmbeddingClassifier
from .mask_utils import MaskDetector
from .models import facenet
from .detect_utils import detect_and_align

# graph = tf.get_default_graph()
def get_threshold(embs, labels, distance='cosine'):
	dist_matrix = cdist(embs, embs, distance)
	sigmas = []

	for i, label in enumerate(labels):
		dist = dist_matrix[i]
		dist = dist[np.where(labels == label)]
		dist = 1 - dist 
		# print(dist)
		
		sigma = dist[np.argmin(dist)]
		sigmas.append(sigma)

	return sigmas

def get_threshold(embs, labels, distance='cosine'):
	dist_matrix = cdist(embs, embs, distance)
	sigmas = []

	for i, label in enumerate(labels):
		dist = dist_matrix[i]
		dist = dist[np.where(labels == label)]
		dist = 1 - dist 
		# print(dist)
		
		sigma = dist[np.argmin(dist)]
		sigmas.append(sigma)

	return sigmas

class FaceRecognizer(object):
	def __init__(self, registration_folder=None, camera_index=0, camera_flip=False, detect_mask=True):
		global facenet
		base_path = os.path.dirname(os.path.realpath(__file__))
		weights_path = os.path.join(base_path, 'model_94k_faces_glintasia_without_norm.hdf5')

		print('[INFO] Loading model ... ')
		facenet.load_weights(weights_path)

		self.clf_no_mask_path = os.path.join(base_path, 'clf_no_mask.pickle')
		self.clf_mask_path = os.path.join(base_path, 'clf_mask.pickle')

		try:
			self.clf = EmbeddingClassifier(registration_folder=registration_folder, mask=False, model_path=self.clf_no_mask_path)
			self.clf_mask = EmbeddingClassifier(registration_folder=registration_folder, mask=True, model_path=self.clf_mask_path)
		except:
			print('[INFO] Not enough idx to create classifier ... ')
		self.camera_index = camera_index
		self.camera_flip = camera_flip
		self.detect_mask = detect_mask

		self.model = tf.keras.models.Model(inputs=facenet.inputs[0], outputs=facenet.get_layer('emb_output').output)
		
		### Change this ###
		self.model_url = "http://localhost:8502/v1/models/arcface_keras:predict"
		self.mask_detector = MaskDetector(base_path)

		if(registration_folder is None):
			self.registration_folder = os.path.join(base_path, 'identities')
		else:
			self.registration_folder = os.path.join(base_path, registration_folder)

		# self.model = facenet 
		self.base_path = os.path.dirname(os.path.realpath(__file__))
		self.embeddings = np.array([])
		self.embeddings_masked = np.array([])
		self.labels = np.array([])
		self.labels_masked = np.array([])

		print('[INFO] Loading identities ... ')
		for i, idx in enumerate(glob.glob(self.registration_folder + '/*')):
			label = os.path.basename(idx)

			npy_path = os.path.join(idx, '%s.npy' % label)
			npy_masked_path = os.path.join(idx, '%s_masked.npy' % label)

			embeddings = np.load(npy_path)
			embeddings_masked = np.load(npy_masked_path)
			labels = np.full(embeddings.shape[0], label)

			if(len(self.embeddings) == 0):
				self.embeddings = embeddings
			else:
				self.embeddings = np.concatenate((self.embeddings, embeddings))
			

			if(len(self.embeddings_masked) == 0):
				self.embeddings_masked = embeddings_masked
			else:
				self.embeddings_masked = np.concatenate((self.embeddings_masked, embeddings_masked))

			self.labels = np.concatenate((self.labels, labels))
			self.labels_masked = np.concatenate((self.labels_masked, labels))

		if(len(np.unique(self.labels)) >= 2):
			self.sigmas = get_threshold(self.embeddings, self.labels)

		if(len(np.unique(self.labels_masked)) >= 2):
			self.sigmas_masked = get_threshold(self.embeddings_masked, self.labels_masked)

	def _histogram_equalization(self, image):
		r, g, b = cv2.split(image)
		r = cv2.equalizeHist(r)
		g = cv2.equalizeHist(g)
		b = cv2.equalizeHist(b)

		rgb = cv2.merge((r, g, b))

		return rgb

	def make_tf_serve_request(self, image):
		data = json.dumps({
			"signature_name":"serving_default",
			"instances":[
				{
					"input_1" : image.tolist()
				}
			]
		})
		headers = {"content-type":"application/json"}
		r = requests.post(self.model_url, data=data, headers=headers)
		predictions = json.loads(r.text)['predictions']
		predictions = np.array(predictions)

		return predictions


	def _face_preprocessing(self, image, size=(170, 170)):
		image = cv2.resize(image, size)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = self._histogram_equalization(image)

		image = (image - 127.5) / 127.5

		return image

	def _is_blur(self, image, threshold=100):
		return cv2.Laplacian(image, cv2.CV_64F).var() < threshold

	def _is_bad_lighting(self, image):
		bright_thres = 0.5
		dark_thres = 0.2
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		dark_part = cv2.inRange(gray, 0, 60)
		bright_part = cv2.inRange(gray, 200, 255)
		# use histogram
		# dark_pixel = np.sum(hist[:30])
		# bright_pixel = np.sum(hist[220:256])
		total_pixel = np.size(gray)
		dark_pixel = np.sum(dark_part > 0)
		bright_pixel = np.sum(bright_part > 0)

		bad_lighting = False
		if dark_pixel/total_pixel > bright_thres:
			bad_lighting = True
		if bright_pixel/total_pixel > dark_thres:
			bad_lighting = True

		return bad_lighting

	def clf_recognize(self, face, masked=False):
		''' Using the clf model to recognize '''
		identity = 'Unknown'
		### Get embeddings from face ###
		# embedding = self.model.predict(np.array([face]))
		embedding = self.make_tf_serve_request(face)
		embedding = embedding / np.linalg.norm(embedding, axis=1).reshape(-1, 1)

		if(not masked):
			label = self.clf.model.predict(embedding)[0]
			probability = self.clf.model.predict_proba(embedding)[0]
			probability = probability[np.argmax(probability)]
		else:
			label = self.clf_mask.model.predict(embedding)[0]
			probability = self.clf_mask.model.predict_proba(embedding)[0]
			probability = probability[np.argmax(probability)]
		# print(label, probability)

		return label, probability

	def recognize(self, face):
		identity = 'Unknown'
		### Get embeddings from face ###
		# embedding = self.model.predict(np.array([face]))
		embedding = self.make_tf_serve_request(face)
		embedding = embedding / np.linalg.norm(embedding, axis=1).reshape(-1, 1)

		### Get the distance matrix ###
		dist_mat = 1 - cdist(embedding, self.embeddings, 'cosine')
		dist_mat = dist_mat[0]
		best_match = np.argmax(dist_mat)

		if(dist_mat[best_match] >= self.sigmas[best_match] + 0.02):
			identity = self.labels[best_match]

		return identity

	def start_standalone_app(self, video=None):
		videoSrc = self.camera_index
		vs = WebcamVideoStream(src=videoSrc).start()

		if(video is not None):
			videoSrc = video 
			vs = cv2.VideoCapture(videoSrc)

		while(True):
			if(video is not None):
				ret, frame = vs.read()
			else:
				frame = vs.read()

			if(video is None and self.camera_flip):
				frame = cv2.flip(frame, flipCode=1)
				frame = cv2.flip(frame, flipCode=0)

			# if(frame is None): continue
			try:
				faces, locations = detect_and_align(frame)
				for face, location in zip(faces, locations):
					bounding_box_color = (0,255,0)
					x1, y1, x2, y2 = location 
					mask = self.mask_detector.predict(face)
					masked = True
					if(mask == 'No Mask'):
						bounding_box_color = (0,0,255)
						masked = False

					cv2.rectangle(frame, (x1, y1), (x2, y2), bounding_box_color, 1)

					### Check image quality ###
					blur_ = self._is_blur(face)
					bad_light_ = self._is_bad_lighting(face)

					label = None
					if(blur_):
						label = 'BLURRY'
					elif(bad_light_):
						label = 'BAD_LIGHTING'
					else:
						if(self.clf is not None):
							label, probability = self.clf_recognize(self._face_preprocessing(face), masked=masked)
							label = '%s - %.2f' % (label, probability)
						else:
							label = self.recognize(self._face_preprocessing(face))

					color = (0,255,0) if (blur_ != True and bad_light_ != True) else (0,0,255) 

					cv2.putText(frame, label, (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)

			except:
				traceback.print_exc(file=sys.stdout)

			cv2.imshow('Frame', frame)
			key = cv2.waitKey(1)
			if(key == ord('q')):
				break 

		vs.stop()
		cv2.destroyAllWindows()

	def deregister(self, name):
		full_path = os.path.join(self.registration_folder, name)
		if(not os.path.exists(full_path)):
			print('[INFO] ID not exists : %s ' % full_path)
		else:
			shutil.rmtree(full_path)
			print('[INFO] ID at %s is removed ... ' % full_path)

	def _register_with_mask(self, name, id_dir, img_dir, masked_dir):
		mask_the_face_folder = os.path.join(self.base_path, 'MaskTheFace')
		mask_types = ['surgical', 'cloth']

		for i, img_file in enumerate(glob.glob(img_dir + '/*.jpg')):
			cmd = "cd {} && python3 mask_the_face.py --path {} --mask_type '{}' --output_dir {}"
			mask_type = mask_types[np.random.randint(0, len(mask_types))]

			cmd = cmd.format(mask_the_face_folder, img_file, mask_type, masked_dir)

			os.system(cmd)

		print('[INFO] Generating embeddings for masked folder ...')
		face_images = []
		for i, img_file in enumerate(glob.glob(img_dir + '/*.jpg')):
			img = cv2.imread(img_file)
			face_images.append(self._face_preprocessing(img))

		embeddings = self.model.predict(np.array(face_images))
		embeddings = embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1) # normalize

		with open(os.path.join(id_dir, '%s_masked.npy' % name), 'wb') as f:
			np.save(f, embeddings)
			print('[INFO] Saved normalized embeddings of masked ID to %s' % os.path.join(id_dir, '%s_masked.npy' % name))

	def register(self, name, video=None):
		videoSrc = self.camera_index
		if(video is not None):
			videoSrc = video 

		print('[INFO] Starting registration ...')
		vid = cv2.VideoCapture(videoSrc)
		start = time.time()
		num_images = 0
		face_images = []
		id_dir = os.path.join(self.registration_folder, name)
		img_dir = os.path.join(id_dir, 'imgs')
		masked_dir = os.path.join(id_dir, 'masked')

		if(not os.path.exists(id_dir)):
			print('[INFO] Creating ID directory ... ')
			os.mkdir(id_dir)

		if(not os.path.exists(img_dir)):
			print('[INFO] Creating images directory ... ')
			os.mkdir(img_dir)

		if(not os.path.exists(masked_dir)):
			print('[INFO] Creating masked images directory ... ')
			os.mkdir(masked_dir)

		while(True):
			ret, frame = vid.read()
			if(video is None and self.camera_flip):
				frame = cv2.flip(frame, flipCode=1)
				frame = cv2.flip(frame, flipCode=0)

			### Detection and alignment ###
			faces, locations = detect_and_align(frame)
			
			for face, location in zip(faces, locations):
				cv2.imshow('Face (Aligned)', face)
				x1, y1, x2, y2 = location 

				blur_ = self._is_blur(face)
				bad_light_ = self._is_bad_lighting(face)

				color = (0,255,0) if (blur_ != True) else (0,0,255) 
				blur_condition = 'Good' if (blur_ != True) else 'Not good'
				cv2.putText(frame, 'Blurriness : %s' % blur_condition, (20,20), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)
				
				color = (0,255,0) if (bad_light_ != True) else (0,0,255) 
				light_condition = 'Good' if (bad_light_ != True) else 'Not good'
				cv2.putText(frame, 'Lighting condition : %s' % light_condition, (20,40), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)

				cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

				if(not blur_ and not bad_light_):
					num_images += 1
					filename   = '%s_%d.jpg' % (name, num_images)
					abs_filename = os.path.join(img_dir, filename)

					cv2.imwrite(abs_filename, face)
					face_images.append(self._face_preprocessing(face))
				else:
					print('[INFO] Skipping bad image ... ')

			cv2.imshow('Frame', frame)
			key = cv2.waitKey(1)
			if(key == ord('q') or time.time() - start >= 5):
				print('[INFO] Stopping registration ... ')
				break

		if(len(face_images) >= 20):
			print('[INFO] Number of images collected : ', len(face_images))
			print('[INFO] Generating embeddings ... ')
			embeddings = self.model.predict(np.array(face_images))
			embeddings = embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1) # normalize

			with open(os.path.join(id_dir, '%s.npy' % name), 'wb') as f:
				np.save(f, embeddings)
				print('[INFO] Saved normalized embeddings to %s' % os.path.join(id_dir, '%s.npy' % name))
		else:
			print('[INFO] Not enough frames registered ... ')

		print('[INFO] Creating masked images folder ...')
		self._register_with_mask(name, id_dir, img_dir, masked_dir)

		print('[INFO] Reinitializing ... ')
		self.__init__()
		vid.release()
		cv2.destroyAllWindows()

		### Rebuild the classifier ###
		os.remove(self.clf_no_mask_path)
		os.remove(self.clf_mask_path)
		try:
			self.clf = EmbeddingClassifier(registration_folder=self.registration_folder, mask=False, model_path=self.clf_no_mask_path)
			self.clf_mask = EmbeddingClassifier(registration_folder=self.registration_folder, mask=True, model_path=self.clf_mask_path)
		except:
			print('[INFO] Not enough idx to create classifier ... ')

	def detect_and_recognize(self, frame):
		global mask_detector
		detections = []
		names = []
		if(self.camera_flip):
				frame = cv2.flip(frame, flipCode=1)
				frame = cv2.flip(frame, flipCode=0)

		try:
			faces, locations = detect_and_align(frame)
			for face, location in zip(faces, locations):
				bounding_box_color = (0,255,0)
				x1, y1, x2, y2 = location 
				
				mask = self.mask_detector.predict(face)
				masked = True
				if(mask == 'No Mask'):
					bounding_box_color = (0,0,255)
					masked = False

				# cv2.rectangle(frame, (x1, y1), (x2, y2), bounding_box_color, 1)

				### Check image quality ###
				blur_ = self._is_blur(face)
				bad_light_ = self._is_bad_lighting(face)

				label = None
				if(blur_):
					label = 'BLURRY'
				elif(bad_light_):
					label = 'BAD_LIGHTING'
				else:
					if(self.clf is not None):
						label, probability = self.clf_recognize(self._face_preprocessing(face), masked=masked)
						label = '%s - %.2f' % (label, probability)
					else:
						label = self.recognize(self._face_preprocessing(face))

				detections.append((x1, y1, x2, y2))
				names.append(label)
		except:
			traceback.print_exc(file=sys.stdout)

		return detections, names
