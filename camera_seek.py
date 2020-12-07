### This is the script for ThermalX - SeekThermal version ###

import os
import glob
import hashlib
import requests
import sys, traceback
import cv2 # 3.4.6.27
# import cvlib as cv --> tensorflow 2.1.0
from face_detection import detect_faces
import time 
import datetime
import pickle
import imutils # any version
import numpy as np # any version
import pandas as pd # any version
import sys, traceback
import face_recognition
import datetime
import tensorflow as tf

# any version of these
# from mtcnn.mtcnn import MTCNN
from playsound import playsound # for warning siren
# from tensorflow.keras import models as models
from scipy.spatial.distance import cosine
from imutils import face_utils
from imutils.video import WebcamVideoStream
### from thermal import * --> For the Terabee version ###
from seek import SeekThermal
from object_tracker import CentroidTracker

### For face recognition ###
from face_recog import get_facenet_model, get_arcface_model, get_embs_from_folder

### for adaptive face recognition ###
from face_recog import face_recog_adaptive as face_recog
from face_recog import get_threshold 

import threading

DPI = 2080
PADDING = 70
CAM_DISTANCE = 60
FOCAL_LENGTH = 5 # / 10 * DPI / 2.54 # mm - 3.34 for the latte panda camera
SENSOR_HEIGHT = 4
KNOWN_HEIGHT = 20 # cm

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
# mask_detector = models.load_model('model_1.h5')
ct = CentroidTracker()
frame_ = None
temps = None
frame_temps = None 

# should switch to predicting the distance difference because
# the offset factor of distance matters less as compared to ratios
# therefore, predicting distance will reduce the risk of mis prediction
body_predictor = pickle.load(open("heat_dist_predictor.h5", "rb"))
amb = 0 # ambient temperature

session = tf.Session()
FRAME_SIZE = 550 # the frame size for display (not processing)
OFFSET_FACTOR = 0.7 # to cure the error when distance/amb is beyond training ds
FAR_OFFSET_FACTOR = 0.8
MIN_TEMP_THRESH = 37.5
CURRENT_LANGUAGE = 'English'
CURRENT_METRIC = 'Celcius'
CASE_ORIENTATION = 'vertical'

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
	known_encodings, known_names = get_embs_from_folder(folders=['faces/', 'masked/'])
else:
	known_encodings, known_names = get_embs_from_folder(folders=None)


ANALYTIC_DATA_DIR = 'analytics/'
analytics_df = []

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

def set_orientation(orientation):
	global CASE_ORIENTATION
	CASE_ORIENTATION = orientation

def get_current_threshold():
	global MIN_TEMP_THRESH
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

	columns = ['Date', 'Time', 'Distance away','Unadjusted_Temp', 'Temperature', 'Overheated','Average ambient temperature']
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
		global CASE_ORIENTATION
		self.vs = WebcamVideoStream(src = 0).start()
		self.seek = SeekThermal()
		self.fr_model = get_facenet_model()

		### for adaptive face recog ###
		self.dist_matrix = get_threshold(known_encodings, known_names)

		self.stopEvent = threading.Event()
		self.warning = threading.Event()
		self.warning_start = 0


		self.app_start = time.time()
		self.total_faces = 0
		
		self.PROCESS_FRAME = True
		self.face_locations, self.face_locations_thermal, self.temperatures, self.distances, self.masks, self.names = [],[],[],[],[],[]

		self.thermal_img = None
		self.normal_img = None
		self.frame_temps = None

		# store the incident ids
		self.pause = 0
		self.breached_pause = 0
		# thermal coordinate system
		if(CASE_ORIENTATION == 'horizontal'):
			### For horizontal ###
			print("[INFO] Horizontal Version ... ")
			self.X_thermal = 100 # 110 for vertical case
			self.Y_thermal = 80 # 60 for vertical case
		else:
			### For Vertical ###
			print("[INFO] Vertical version ... ")
			self.X_thermal = 99
			self.Y_thermal = 82
		self.W_thermal = (360 - 2*PADDING - 10)
		self.H_thermal = (360 - 2*PADDING)

		# make a data folder for today if the folder is not
		# already exist
		now = datetime.datetime.now()
		today = str(now.day) + "-" + str(now.month) + "-" + str(now.year)

		self.data_dir = 'data/' + today + "/"
		if(not os.path.exists(self.data_dir)):
			os.mkdir(self.data_dir)

		### Run the seek thermal app in the background ###
		t_thermal = threading.Thread(target=os.system, args = ("sudo ./seekware-simple > thermal_log.log",))
		t_thermal.daemon = True
		t_thermal.start()

		print("[INFO] Booting up seek thermal script ... ")
		time.sleep(2.0) ### Sleep for thermal app to boot up ###

	def __del__(self):
		# stopping camera stream
		global current_incidents
		global incident_ids
		global MIN_TEMP_THRESH

		MIN_TEMP_THRESH = 37.5 # reset the minimum temperature threshold

		self.vs.stop()
		
		# refresh current incidents
		current_incidents = []

		# refreshes incident idexes
		incident_ids = []

		# save the last recorded data
		# so when the user close the browser and it has not been 1000 rows recorded yet
		# the remaining data will be saved anyway
		columns = ['Date', 'Time', 'Distance away','Unadjusted_Temp', 'Temperature', 'Overheated','Average ambient temperature']
		data_frame = pd.DataFrame(df, columns = columns)

		file_name = self.data_dir + str(time.time()) + ".csv"
		data_frame.to_csv(file_name)
		self.stopEvent.set()

		print("________________________________________________________________________")
		print("[INFO] Camera stream stopped ... ")

		print("[INFO] Uploading data file to server ...")
		data = {'password' : hashlib.md5("HieuDepTry".encode()).hexdigest()}
		files = {'file' : open(file_name, "r")}

		r = requests.post("http://167.71.193.193:8080/upload", data=data, files=files)
		print(r.text)

		# reset the incident list
		files = glob.glob('static/img/incidents/*.jpg')
		for f in files:
			os.remove(f)

		os.system("sudo killall seekware-simple")

	def __to_fahrenheit(self, celcius):
		f = (celcius * 9/5) + 32

		return f

	def save_data(self, df):
		global files_list
		# save the records
		print(df)
                
		columns = ['Date', 'Time', 'Distance away','Unadjusted_Temp', 'Temperature', 'Overheated','Average ambient temperature']
		data_frame = pd.DataFrame(df, columns = columns)
		file_name = self.data_dir + str(time.time()) + '.csv'
		data_frame.to_csv(file_name)
		files_list.append(file_name)

	### A fix for the corner visible at the thermal image ###
	def zoom(self, img, percentage=0.80):
		height, width = img.shape[:2]
		padding_height = int(((1 - percentage) * height) / 2) 
		padding_width = int(((1 - percentage) * width) / 2)

		img_cropped = img[padding_height:height-padding_height, padding_width: width - padding_width]
		img = cv2.resize(img_cropped, (width, height))

		return img

	def get_frame(self):
		global amb_list
		global people_per_hour
		global incidents_list
		global camera_view
		global df
		global incident_ids 
		global frame_

		try:
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
																
				frame = self.vs.read()
				frame = cv2.resize(frame, (400, 400))				

				frame = cv2.flip(frame, flipCode = 0)
				frame = cv2.flip(frame, flipCode = 1)
				cv2.rectangle(frame, (self.X_thermal, self.Y_thermal), (self.X_thermal + self.W_thermal, self.Y_thermal+self.H_thermal), (255,0,0), 1)
																
				# grab the frame dimensions and convert it to a blob
				(H, W) = frame.shape[:2]
				cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
															 
				# pass the blob through the network and obtain the detections and
				# predictions
				#net.setInput(blob)
				#detections = net.forward()
				detections, identities = face_recog((known_encodings, known_names), frame, self.fr_model, self.dist_matrix)
				data = []
				try:
					frame_, temps = self.seek.get_frame()
					frame_, temps = self.zoom(frame_), self.zoom(temps) ### default zoom rate is 0.80 ###

					# temps = cv2.flip(temps, flipCode= -1)

					### Getting the ambient temperature ###
					percentile_50 = np.percentile(temps, 15)
					percentile_75 = np.percentile(temps, 30)
					ambs = temps[np.where(temps < percentile_75)]
					ambs = ambs[np.where(ambs > percentile_50)]
					amb = np.mean(ambs)
					amb_list.append(round(amb,2))
					amb_list_x_axis.append(get_time_for_plot())

					### if the ambient temp list is more than 100 items, remove first one ###
					if(len(amb_list) > 100):
						amb_list.pop(0)
						amb_list_x_axis.pop(0)

					### round temperature value to integer ###
					self.frame_temps = temps.astype(np.uint8)
						
				except Exception as e:
					traceback.print_exc(file=sys.stdout)
					print("[INFO] Thermal camera crashed, re-running ...")					
					pass  
				if(self.PROCESS_FRAME):
					self.face_locations,self.face_locations_thermal, self.temperatures, self.distances, self.masks, self.names = [],[],[],[],[],[]
					
					for i, detection in enumerate(detections):
						incident = []
						### Add identity ###
						self.names.append(identities[i])
						now = datetime.datetime.now()
						incident.append(str(now.day) + "/" + str(now.month) + "/" + str(now.year))
						incident.append(str(now.hour) + ":" + str(now.minute) + ":" + str(now.second)) 

						### if face is detected then record in people frequency data ###
						self.total_faces += 1
						hours = (time.time() - self.app_start) / 3600
						frequency = int(self.total_faces / (hours * 10)) # approximately 10 frames/person
						people_per_hour.append(frequency)
						people_per_hour_x_axis.append(get_time_for_plot())

						### if people per hour array is more than 100 items, remove one ###
						if(len(people_per_hour) >= 100):
							people_per_hour.pop(0)
							people_per_hour_x_axis.pop(0)
					
						# box = detections[0,0,i,3:7] * np.array([W,H,W,H])
						(startX, startY, endX, endY) = detection #box.astype("int")

						### if the startX and startY is out of the box limit, not a correct detection ###
						if(startX > 400 or startY > 400): continue
 
						if(startX > self.X_thermal and startY > self.Y_thermal and endX < self.X_thermal+self.W_thermal and endY < self.Y_thermal + self.H_thermal):
							self.face_locations.append((startX, startY, endX, endY))
						(x,y,w,h) = startX, startY, endX - startX, endY-startY#face_utils.rect_to_bb(rect) # rect['box']
						

						try:
							# disActivatetance = (focal_length * known_height * img_height) / (object_height(px) * sensor_height)
							distance = (FOCAL_LENGTH * KNOWN_HEIGHT * H)/(h*SENSOR_HEIGHT) #2 * (FOCAL_LENGTH * KNOWN_HEIGHT) / h
							incident.append("{0:.2f}".format(distance))
							display_dist = distance
							self.distances.append(display_dist)

							
							### frame_temps from (150,200) to (400,400)
							self.frame_temps = cv2.resize(self.frame_temps, (400,400))

							### Locate the face coords in the thermal image ###
							new_X, new_Y, new_W, new_H = 0,0,0,0
							if(x - self.X_thermal > 0 and y - self.Y_thermal > 0):
								new_X = max(int((x - self.X_thermal)/self.W_thermal * 400),0)
								new_Y = max(int((y - self.Y_thermal)/self.H_thermal * 400),0)
								new_W = int((w/self.W_thermal * 400))+10
								new_H = int((h/self.H_thermal * 400))+20
								# cv2.rectangle(self.frame_temps, (new_X, new_Y), (new_X+new_W, new_Y+new_H), (0,0,0),1)
							else: continue
							
							### get the thermal region of the face ###
							self.face_locations_thermal.append((new_X, new_Y, new_W, new_H))
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
								incident.append("{0:.2f}".format(mean))

								# just for sure, the machine learnig model was trained on a caliberation
								# based on a device with base line temp = 34
								# so if we test this code on device with base line > 34
								if(mean > 34.5): mean -= 1.0

								### Check if real distance yields better result
								prediction =  body_predictor.predict(np.array([[distance, amb]]))
									
								ratio = prediction[0]
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
																
								# self.temperatures.append(mean)
								if(distance > 220):
									self.temperatures.append(36.3)
								else:
									self.temperatures.append(mean)
								incident.append("{0:.2f}".format(mean))

							except:
								self.temperatures.append(36.5)
							
							# incident.append("{0:.2f}".format(self.temperatures[len(self.temperatures) - 1]))

							if(self.temperatures[len(self.temperatures) - 1] > MIN_TEMP_THRESH):
								if(not HOT_OBJECT): 
									if(self.breached_pause == 0 and not self.pause):
										self.breached_pause = 1
										t_warn = threading.Thread(target=playsound, args=("audio/beep-06.mp3",))
										t_warn.start()

										audio_file_name = 'audio/breached.mp3'

										if(CURRENT_LANGUAGE == 'Chinese'):
											audio_file_name = 'audio/breached_chinese.mp3'
										elif(CURRENT_LANGUAGE == 'Japanese'):
											audio_file_name = 'audio/breached_japanese.mp3'

										t_vocal = threading.Thread(target=playsound, args=(audio_file_name,))
										t_vocal.start()
										incident.append("Yes")
									else:
										incident.append("pause")
								else:
									incident.append("hot_object")	
							else:
								incident.append("No")

							incident.append("{0:.2f}".format(np.mean(amb_list)))
							df.append(incident)
						except:
							traceback.print_exc(file=sys.stdout)

					### end for ###		

				### Update centroid tracker ###
				rects, objects = ct.update(self.face_locations)
				self.PROCESS_FRAME = not self.PROCESS_FRAME

				for (objectID, centroid), (startX,startY,endX,endY), (new_X, new_Y, new_W, new_H), temp, dist, name in zip(objects.items(), self.face_locations, self.face_locations_thermal, self.temperatures, self.distances, self.names):
					cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0),1)
					cv2.rectangle(frame_, (new_X, new_Y), (min(new_X + new_W, 400), min(new_Y + new_H, 400)), (0,0,0), 2)
					### integrate face recognition ###
					if(CURRENT_METRIC == 'Celcius'):
						cv2.putText(frame, "{0:.1f}.C".format(temp) , (startX,startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
					else:
						cv2.putText(frame, "{0:.1f}.F".format(self.__to_fahrenheit(temp)), (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255) ,1)
					cv2.putText(frame, "Distance {0:.1f}".format(dist) , (startX,endY+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
					cv2.putText(frame, "Identity : {}".format(name), (startX, endY+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

					if(objectID not in incident_ids):
						#check all locations in the list
						for i,(startX, startY, endX, endY) in enumerate(self.face_locations):

							# if the location include the centroid
							if(startX < centroid[0] and endX > centroid[0] and startY < centroid[1] and endY > centroid[1]):
								incident_ids.append(objectID)
								img = frame[max(0,startY-40):min(endY+40,400),max(0,startX-40):min(400,endX+40)]
								img = cv2.resize(img, (250,250))
								temperature = self.temperatures[i]
								distance = self.distances[i]

								file_header = "{0:.1f}".format(temperature)
								file_header += "_" + "{0:.1f}".format(distance)

								now = datetime.datetime.now()
								checkin = "_" + str(("%02d" % now.hour) + ":" + ("%02d" % now.minute) + " - " + ("%02d" % now.day) + "-" + ("%02d" % now.month) + "-" + ("%02d" % now.year))

								file_name = file_header + str(checkin) + "_.jpg"
								incidents_list.append(file_name)

								if(temperature >= MIN_TEMP_THRESH):
									if(len(current_incidents) < 6):
										current_incidents.append(file_name)
									else:
										current_incidents.pop(0) # remove the first incident in the list
										current_incidents.append(file_name)

								cv2.imwrite("static/img/incidents/" + file_name, img)

				try:					
					### A Fix for the seek thermal mounting problem ###
					(H_, W_) = frame.shape[:2]
					frame = cv2.resize(frame, (W_, H_))
					frame = cv2.resize(frame, (FRAME_SIZE + 100, FRAME_SIZE))
					
					
					self.normal_img = frame 					
					
					frame_ = cv2.resize(frame_, (FRAME_SIZE + 100, FRAME_SIZE))

					self.thermal_img = frame_
				except:
					traceback.print_exc(file=sys.stdout)
					print("[INFO] Drop frame")
		except Exception as e: 
			traceback.print_exc(file=sys.stdout)
			print("[INFO] Programme ended")
			raise e
		
		if(camera_view == "GRAY"):
			gray = cv2.cvtColor(self.thermal_img, cv2.COLOR_BGR2GRAY)
			self.thermal_img[:,:,0] = gray
			self.thermal_img[:,:,1] = gray
			self.thermal_img[:,:,2] = gray 

		
		image = self.normal_img
		ret, jpeg = cv2.imencode('.jpg', image)

		return jpeg.tobytes()	
