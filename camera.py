### This is the ThermalX - Terabee version ###

import os
import glob
import sys, traceback
import cv2 # 3.4.6.27
import cvlib as cv
import time 
import datetime
import pickle
import imutils # any version
import numpy as np # any version
import pandas as pd # any version
import sys, traceback
import face_recognition
import datetime

# any version of these
# from mtcnn.mtcnn import MTCNN
from playsound import playsound # for warning siren
from tensorflow.keras import models as models
from scipy.spatial.distance import cosine
from imutils import face_utils
from imutils.video import WebcamVideoStream
from thermal import *
from object_tracker import CentroidTracker

import threading

DPI = 2080
PADDING = 70
CAM_DISTANCE = 60
FOCAL_LENGTH = 5 # / 10 * DPI / 2.54 # mm - 3.34 for the latte panda camera
SENSOR_HEIGHT = 4
KNOWN_HEIGHT = 20 # cm
# DPI = 96

# Loading the color map
r = []
g = []
b = []

with open('colormap.txt', 'r') as f:
		for i in range(256):
				x,y,z = f.readline().split(',')
				r.append(x)
				g.append(y)
				b.append(z.replace(";\n", ""))

colormap = np.zeros((256, 1, 3), dtype=np.uint8)
colormap[:, 0, 0] = b
colormap[:, 0, 1] = g
colormap[:, 0, 2] = r


net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "dnn_model.caffemodel")
mask_detector = models.load_model('model_1.h5')
ct = CentroidTracker()

frame_ = None
temps = None
frame_temps = None 

# should switch to predicting the distance difference because
# the offset factor of distance matters less as compared to ratios
# therefore, predicting distance will reduce the risk of mis prediction
body_predictor = pickle.load(open("heat_dist_predictor.h5", "rb"))
amb = 0 # ambient temperature

FRAME_SIZE = 550 # the frame size for display
OFFSET_FACTOR = 0.7 # to cure the error when distance/amb is beyond training ds
FAR_OFFSET_FACTOR = 0.8
MIN_TEMP_THRESH = 37.5
CURRENT_LANGUAGE = 'English'
CURRENT_METRIC = 'Celcius'

# Recording interval :
# For every one hour there will be one record time
# each record will include : 
# Time From - To (broken down to 10 minutes intervals)
# Average ambiant temperature
# Number of incidents
# Number of overheated incidents
# file format thermal_data_<date_time>.csv
RECORDING_INTERVAL = 1 * 10 * 60 # one hour

def lumination_correct(img):
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)

	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize = (8,8))
	cl = clahe.apply(l)

	limg = cv2.merge((cl,a,b))
	final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

	return final

# add on feature over here
# so in this one we need to return 
# 1. who this guy is
# 2. whether he is wearing a mask or not
# 3. and how many percent confidence
FACE_DIR = "faces/"

known_encodings = []
known_names = []

# first load all known faces into an array
if(not os.path.exists('validation_encodings.pickle') or not os.path.exists('known_names.pickle')):
	for (dir, dirs, files) in os.walk(FACE_DIR):
		for file in files:
			img = cv2.imread(FACE_DIR + file)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			(h, w) = img.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
			net.setInput(blob)
			detections = net.forward()
			# img = lumination_correct(img)
			# img = lumination_correct(img)

			# extract the face region
			#locations = face_recognition.face_locations(img)
			if(len(detections) < 1):
				continue
			else:
				for i in range(0, detections.shape[2]):
					if(detections[0,0,i,2] < 0.5):
						continue

					print("Face detected at " + (FACE_DIR + file))
					box = detections[0,0,i,3:7] * np.array([w,h,w,h])
					(startX, startY, endX, endY) = box.astype("int")

					top = startY
					right = endX
					bottom = endY
					left = startX
					# (top, right, bottom, left) = locations[0]
					(x,y,w,h) = left, top, right - left, bottom - top

					# img = cv2.resize(img, (0,0), fx = 0.25, fy = 0.25)
					encodings = face_recognition.face_encodings(img, [(top, right, bottom, left)])
					if(len(encodings) < 1):
						continue
					else:
						known_encodings.append(encodings[0])
						known_names.append(file.split(".")[0])

	pickle.dump(known_encodings, open("validation_encodings.pickle", "wb"))
	pickle.dump(known_names, open("known_names.pickle", "wb"))

else:
	known_encodings = pickle.load(open("validation_encodings.pickle", "rb"))
	known_names = pickle.load(open("known_names.pickle", "rb"))

known_encodings = np.array(known_encodings)
ANALYTIC_DATA_DIR = 'analytics/'
analytics_df = []

def recognize(face):
	global analytics_df
	# constants are defaulted for mask wearers
	try:
		if(len(analytics_df) >= 100):
			anal_df = pd.DataFrame(data = analytics_df, columns=['distance','cosine_similarity','verfied_as'])
			analytics_df = []

			anal_df.to_csv(ANALYTIC_DATA_DIR + "/" + str(time.time()) + ".csv")

		name = "Unknown"

		'''
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		confidence = 0
		(h,w) = face.shape[:2]
		TOLERANCE = 0.60
		COSINE_SIMILARITY = 0.93
		''' 

		mask = mask_detector.predict(np.array([cv2.resize(face, (224,224))]).reshape(1,224,224,3))
		index = np.argmax(mask[0])

		mask = 'With Mask'
		if(index == 1):
			mask = 'Without Mask'
			# TOLERANCE = 0.4
			# COSINE_SIMILARITY = 0.965


		'''
		face_encoding = face_recognition.face_encodings(face, [(0,w,h,0)])
		matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
		face_distances = face_recognition.face_distance(known_encodings, face_encoding)

		best_match = np.argmin(face_distances)


		if(matches[best_match]):
			cosine_similarity = 1 - cosine(known_encodings[best_match], face_encoding)
			#print(cosine_similarity)
			analytics_df.append([face_distances[best_match], cosine_similarity, known_names[best_match]])
			if(cosine_similarity >= COSINE_SIMILARITY):
				name = known_names[best_match]
				#print(name)
				confidence = cosine_similarity
			else:
				name = "Unknown"
		else:
			name = "Unknown"
		'''

		return name, mask, confidence
	except:
		return None, None, None

# oh, actually this too
amb_list = []
amb_list_x_axis = []

# terrence said that this one should have time unit
# in the x axis
people_per_hour = []
people_per_hour_x_axis = []
incidents_list = []
incident_ids = [] # to count number of people passing by
current_incidents = [] # to retrieve top 4-5 incidents (most current incidents)

camera_view = "RGB"

app_start = datetime.datetime.now()

def set_camera_view(cam_view):
	global camera_view
	camera_view = cam_view

def set_temp_threshold(threshold):
	global MIN_TEMP_THRESH
	MIN_TEMP_THRESH = threshold

def set_current_language(language):
	global CURRENT_LANGUAGE
	CURRENT_LANGUAGE = language

def set_current_metric(metric):
	global CURRENT_METRIC
	CURRENT_METRIC = metric

def get_current_threshold():
	return MIN_TEMP_THRESH

def get_amb():
	return amb_list, amb_list_x_axis

def get_average_amb():
	average_amb = np.mean(amb_list)
	average_amb = "{0:.1f}".format(average_amb)

	return average_amb

def get_people_per_hour():
	return people_per_hour, people_per_hour_x_axis

def get_incidents_list():
	return incidents_list

def get_current_incidents():
	return current_incidents

def get_people_count():
	return len(incident_ids)

files_list = []
df = []
def move_file_to_static():
	# move file to static folder then delete it
	# need to trigger a command to flush all existing data as well
	root_data = pd.DataFrame()
	now = datetime.datetime.now()

	time = str(now.day) + '-' + str(now.month) + '-' + str(now.year)
	from_ = str(app_start.hour) + ':' + str(app_start.minute) 
	to_ = str(now.hour) + ':' + str(now.minute)

	columns = ['Date', 'Time', 'Distance away', 'Temperature', 'Overheated','Average ambient temperature']
	remaining_data = pd.DataFrame(df, columns = columns)
	
	for i, file in enumerate(files_list):
		data = pd.read_csv(file, header = 0)

		if(i == 0):
			root_data = data 
		else:
			root_data = pd.concat([root_data, data])

	root_data = pd.concat([root_data, remaining_data])

	file_name = 'data_' + time + "_" + from_ + '_' + to_ + '.csv' 
	root_data.to_csv("static/" + file_name)

	return file_name

def get_time_for_plot():
	now = datetime.datetime.now()

	hour = now.hour 
	minute = now.minute 
	second = now.second 

	time = str(hour) + ":" + str(minute) + ":" + str(second)

	return time 

class Camera(object):
	def __init__(self):
		self.vs = WebcamVideoStream(src = 0).start()
		self.evo = EvoThermal()

		self.stopEvent = threading.Event()
		self.warning = threading.Event()
		self.warning_start = 0


		self.app_start = time.time()
		self.total_faces = 0
		#self.amb_list = []	
		#self.people_per_hour = []
		
		self.PROCESS_FRAME = True
		self.face_locations, self.temperatures, self.distances, self.masks, self.names = [],[],[],[],[]

		self.thermal_img = None
		self.normal_img = None
		self.frame_temps = None

		# store the incident ids
		# self.incident_ids = []
		self.pause = 0
		self.breached_pause = 0
		# thermal coordinate system
		self.X_thermal = 40
		self.Y_thermal = 60
		self.W_thermal = (400 - 2*PADDING)
		self.H_thermal = (400 - 2*PADDING + 60)

		print("[INFO] Starting thermal app ... ")
		time.sleep(3.0)
		#self.t_thermal = threading.Thread(target=os.system, args=("./seekware-simple"))
		#self.t_thermal.daemon = True
		#self.t_thermal.start()


		# make a data folder for today if the folder is not
		# already exist
		now = datetime.datetime.now()
		today = str(now.day) + "-" + str(now.month) + "-" + str(now.year)

		self.data_dir = 'data/' + today + "/"
		if(not os.path.exists(self.data_dir)):
			os.mkdir(self.data_dir)

	def __del__(self):
		# stopping camera stream
		global current_incidents
		global incident_ids
		global MIN_TEMP_THRESH

		MIN_TEMP_THRESH = 35.8 # reset the minimum temperature threshold

		self.vs.stop()
		
		# refresh current incidents
		current_incidents = []

		# refreshes incident idexes
		incident_ids = []

		# save the last recorded data
		# so when the user close the browser and it has not been 1000 rows recorded yet
		# the remaining data will be saved anyway
		columns = ['Date', 'Time', 'Distance away', 'Temperature', 'Overheated','Average ambient temperature']
		data_frame = pd.DataFrame(df, columns = columns)
		data_frame.to_csv(self.data_dir + str(time.time()) + ".csv")
		self.stopEvent.set()
		print("[INFO] Camera stream stopped ... ")

		# reset the incident list
		files = glob.glob('static/img/incidents/*.jpg')
		for f in files:
			os.remove(f)

		#self.t_stop_thermal = threading.Thread(target=os.system, args=("killall seekware-simple"))
		#self.t_stop_thermal.start()

	def save_data(self, df):
		global files_list
		# save the records
		columns = ['Date', 'Time', 'Distance away', 'Temperature', 'Overheated','Average ambient temperature']
		data_frame = pd.DataFrame(df, columns = columns)
		file_name = self.data_dir + str(time.time()) + '.csv'
		data_frame.to_csv(file_name)
		files_list.append(file_name)

	def celcius_to_fahrenheit(self, c):
		return ( c * 9/5 ) + 32

	def get_frame(self):
		global amb_list
		global people_per_hour
		global incidents_list
		global camera_view
		global df
		global incident_ids 
		# df = []
							
		try:
			#PROCESS_FRAME = True
			if(self.pause):
				self.pause += 1
				if(self.pause > 30) : self.pause = 0

			if(self.breached_pause):
				self.breached_pause += 1
				if(self.breached_pause > 30): self.breached_pause = 0

			if not self.stopEvent.is_set():
				if(len(df) >= 100 and len(os.listdir(self.data_dir)) < 1000):
					# save the records
					t_save = threading.Thread(target=self.save_data, args = (df,))
					t_save.start()

					# reset data frame
					df = []

				# loop_start = time.time()
																
				frame = self.vs.read()
				frame = cv2.resize(frame, (400, 400))

				frame = cv2.flip(frame, flipCode = 0)
				frame = cv2.flip(frame, flipCode = 1)
				cv2.rectangle(frame, (self.X_thermal, self.Y_thermal), (self.X_thermal + self.W_thermal, self.Y_thermal+self.H_thermal), (255,0,0), 2)
																
				# grab the frame dimensions and convert it to a blob
				(H, W) = frame.shape[:2]
				#blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
															 
				# pass the blob through the network and obtain the detections and
				# predictions
				#net.setInput(blob)
				#detections = net.forward()
				detections, confidences = cv.detect_face(lumination_correct(frame))
				data = []
				try:
					if(not self.evo.broken):
						data, temps, frame_ = self.evo.run()
						percentile_50 = np.percentile(temps, 15)
						percentile_75 = np.percentile(temps, 30)
						ambs = temps[np.where(temps < percentile_75)]
						ambs = ambs[np.where(ambs > percentile_50)]
						amb = np.mean(ambs)
						#print(self.amb_list)
						amb_list.append(round(amb,2))
						amb_list_x_axis.append(get_time_for_plot())
						if(len(amb_list) > 100):
							amb_list.pop(0)
							amb_list_x_axis.pop(0)


						temps = cv2.flip(temps, flipCode=0)
						frame_ = np.round(data, 0).astype(np.uint8)
						frame_ = cv2.applyColorMap(frame_, colormap)
						frame_ = cv2.flip(frame_, flipCode=0)

						self.frame_temps = temps.astype(np.uint8)
						#self.frame_temps = cv2.flip(self.frame_temps, flipCode = 1)
					else:
						self.evo = EvoThermal()
						
				except (Exception, serial.serialutil.SerialException) as e:
					traceback.print_exc(file=sys.stdout)
					print("[INFO] Thermal camera crashed, re-running ...")
					self.evo = EvoThermal()
					

				if(self.PROCESS_FRAME):
					self.face_locations, self.temperatures, self.distances, self.masks, self.names = [],[],[],[],[]
					for confidence, detection in zip(confidences, detections):
						incident = []
						now = datetime.datetime.now()
						incident.append(str(now.day) + "/" + str(now.month) + "/" + str(now.year))
						incident.append(str(now.hour) + ":" + str(now.minute) + ":" + str(now.second)) 

						# confidence = detections[0,0,i,2]
						if(confidence < 0.450):
							continue

						self.total_faces += 1
						hours = (time.time() - self.app_start) / 3600
						frequency = int(self.total_faces / (hours * 10)) # approximately 10 frames/person
						people_per_hour.append(frequency)
						people_per_hour_x_axis.append(get_time_for_plot())
						if(len(people_per_hour) >= 100):
							people_per_hour.pop(0)
							people_per_hour_x_axis.pop(0)
					
						# box = detections[0,0,i,3:7] * np.array([W,H,W,H])
						(startX, startY, endX, endY) = detection #box.astype("int")
						if(startX > 400 or startY > 400): continue

						if(startX > self.X_thermal and startY > self.Y_thermal and endX < self.X_thermal+self.W_thermal and endY < self.Y_thermal + self.H_thermal):
							self.face_locations.append((startX, startY, endX, endY))
						(x,y,w,h) = startX, startY, endX - startX, endY-startY#face_utils.rect_to_bb(rect) # rect['box']
						# face = frame[y:y+h,x:x+w]

						# name, mask, confidence = recognize(face)
						#if(name):
						#	self.names.append(name)
						#	self.masks.append(mask)
						#else:
						#	self.names.append("Unknown")
						#	self.masks.append("Without Mask")
						# print(name)

						try:
							# disActivatetance = (focal_length * known_height * img_height) / (object_height(px) * sensor_height)
							distance = (FOCAL_LENGTH * KNOWN_HEIGHT * H)/(h*SENSOR_HEIGHT) #2 * (FOCAL_LENGTH * KNOWN_HEIGHT) / h
							incident.append("{0:.2f}".format(distance))
							display_dist = distance
							self.distances.append(display_dist)
							if(distance > 100):
								distance  = 100
							elif(distance < 70):
								distance = 70

							self.frame_temps = cv2.resize(self.frame_temps, (400,400))

							# cv2.rectangle(frame_temps, (x,y+10),(x+w,y+10+h), (0,0,0),2)
							new_X, new_Y, new_W, new_H = 0,0,0,0
							if(x - self.X_thermal > 0 and y - self.Y_thermal > 0):
								new_X = max(int((x - self.X_thermal)/self.W_thermal * 400),0)
								new_Y = max(int((y - self.Y_thermal)/self.H_thermal * 400),0)
								new_W = int((w/self.W_thermal * 400))+10
								new_H = int((h/self.H_thermal * 400))+20
								cv2.rectangle(self.frame_temps, (new_X, new_Y), (new_X+new_W, new_Y+new_H), (0,0,0),2)
							else: continue
							temps = self.frame_temps[new_Y:min(new_Y+new_H, 400), new_X: min(new_X + new_W,400)].flatten()

							# check the standard deviation
							# if greater than 2 -> hot object detected
							stdeviation = np.std(temps)

							HOT_OBJECT = False

							# still got error of empty axes
							try:
								_percentile_50 = np.percentile(temps,90)
								where = np.where(temps > _percentile_50)
								mean = np.mean(temps[where])

								# just for sure, the machine learnig model was trained on a caliberation
								# based on a device with base line temp = 34
								# so if we test this code on device with base line > 34
								if(mean > 34.5): mean -= 1.0

								### Check if real distance yields better result
								prediction =  body_predictor.predict(np.array([[distance, amb]]))
									
								ratio = prediction[0]
								# print(mean, ratio)
								mean = mean + ratio 

								# config the baseline
								if(int(mean) % 30 < 6 or int(mean) < 30):
									add = 36 - (int(mean) % 36)
									mean = mean + add
											
								# variance/bias correction (based on statistics)
								if(mean > 37.5):
									mean -= FAR_OFFSET_FACTOR

								if(mean < 36.2):
									mean += OFFSET_FACTOR

								if(not HOT_OBJECT and not self.pause):
									# detects abnormally high temperature and uneven
									# distribution of temperature -> hot object
									if(stdeviation > 2 and mean > 37.8):
										self.pause = 1
										print("[INFO] Hot object detected ... ")
										HOT_OBJECT = True

										t_hot_obj = threading.Thread(target=playsound, args=('audio/hot_object.mp3',))
										t_hot_obj.start()
																
								self.temperatures.append(mean)
							except:
								self.temperatures.append(36.5)
							
							incident.append("{0:.2f}".format(self.temperatures[len(self.temperatures) - 1]))

							if(self.temperatures[len(self.temperatures) - 1] > MIN_TEMP_THRESH):
								if(not HOT_OBJECT): 
									if(self.breached_pause == 0 and not self.pause):
										self.breached_pause = 1
										print("[INFO] AHIHI")

										audio_file_name = 'audio/breached.mp3'

										if(CURRENT_LANGUAGE == 'Chinese'):
											audio_file_name = 'audio/breached_chinese.mp3'
										elif(CURRENT_LANGUAGE == 'Japanese'):
											audio_file_name = 'audio/breached_japanese.mp3'

										t_warn = threading.Thread(target=playsound, args=("audio/beep-06.mp3",))
										t_warn.start()

										t_vocal = threading.Thread(target=playsound, args=(audio_file_name,))
										t_vocal.start()
										incident.append("Yes")
									
							else:
								incident.append("No")

							incident.append("{0:.2f}".format(np.mean(amb_list)))
							df.append(incident)
						except:
							traceback.print_exc(file=sys.stdout)

					### end for ###		
				rects, objects = ct.update(self.face_locations)
				self.PROCESS_FRAME = not self.PROCESS_FRAME

				for (objectID, centroid), (startX,startY,endX,endY), temp, dist in zip(objects.items(), self.face_locations, self.temperatures, self.distances):
					cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0),2)

					cv2.putText(frame, "{0:.1f}".format(temp) , (startX,startY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
					cv2.putText(frame, "Distance {0:.1f}".format(dist) , (startX,endY+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
					#cv2.putText(frame, name , (endX,startY+40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
					# cv2.putText(frame, mask , (startX,endY+40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)

					if(objectID not in incident_ids):
						#check all locations in the list
						for i,(startX, startY, endX, endY) in enumerate(self.face_locations):

							# if the location include the centroid
							if(startX < centroid[0] and endX > centroid[0] and startY < centroid[1] and endY > centroid[1]):
								incident_ids.append(objectID)
								img = frame[max(0,startY-40):min(endY+40,400),max(0,startX-40):min(400,endX+40)]
								print(max(0,startY-40),min(endY+40,400),max(0,startX-40),min(400,endX+40))
								img = cv2.resize(img, (250,250))
								temperature = self.temperatures[i]
								distance = self.distances[i]
								# mask = self.masks[i]
								# identity = self.names[i]

								file_header = "{0:.1f}".format(temperature)
								file_header += "_" + "{0:.1f}".format(distance)
								# file_header += "_" + mask 
								# file_header += "_" + identity

								checkin = "_" + str(("%02d" % now.hour) + ":" + ("%02d" % now.minute) + " - " + ("%02d" % now.day) + "-" + ("%02d" % now.month) + "-" + ("%02d" % now.year))

								file_name = file_header + str(checkin) + "_.jpg"
								incidents_list.append(file_name)

								if(len(current_incidents) < 6):
									current_incidents.append(file_name)
								else:
									current_incidents.pop(0) # remove the first incident in the list
									current_incidents.append(file_name)

								cv2.imwrite("static/img/incidents/" + file_name, img)


				# self.alert_pane.draw()
				# self.alert_pane.img_list = []
				try:
					#frame = cv2.resize(self.frame_temps, (FRAME_SIZE + 100, FRAME_SIZE))
					frame = cv2.resize(frame, (FRAME_SIZE + 100, FRAME_SIZE))
					self.normal_img = frame 
					
					#frame_ = cv2.resize(self.frame_temps, (FRAME_SIZE + 100, FRAME_SIZE))
					frame_ = cv2.resize(frame_, (FRAME_SIZE + 100, FRAME_SIZE))
					self.thermal_img = frame_
				except:
					print("[INFO] Drop frame")
				# print(time.time() - loop_start)
		except Exception as e: 
			# self.vs.stop()
			print("[INFO] Programme ended")
			raise e
		
		if(camera_view == "GRAY"):
			# print("[INFO] Camera view changed to gray ... ")
			gray = cv2.cvtColor(self.thermal_img, cv2.COLOR_BGR2GRAY)
			self.thermal_img[:,:,0] = gray # cv2.cvtColor(self.thermal_img, cv2.COLOR_BGR2GRAY)
			self.thermal_img[:,:,1] = gray
			self.thermal_img[:,:,2] = gray 

		image = cv2.hconcat([self.normal_img, self.thermal_img])
		ret, jpeg = cv2.imencode('.jpg', image)

		return jpeg.tobytes()	
