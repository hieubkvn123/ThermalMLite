import os
import cv2
import numpy as np
import pandas as pd
import time
import csv
import sys, traceback
import multiprocessing 
from PIL import Image,ImageOps
from matplotlib import cm


class SeekThermal():
	def __init__(self, width=400, height=400):
		self.pixd = np.zeros([150,200], dtype = np.float32)
		self.pixmax = 40
		self.pixmin = 20

		self.width = width
		self.height = height

		self.SEEK_WIDTH = 200
		self.SEEK_HEIGHT = 150

		self.variable = 1 / (self.pixmax - self.pixmin)

		self.current_frame = np.zeros((self.height, self.width, 3))
		self.current_data  = np.zeros((self.height, self.width))

	def get_frame(self, width = 400, height = 400):
		reader = None
		try:
			# start = time.time()
			thermograph = open("thermography.csv",'r')
			reader = list(csv.reader(thermograph))

			if(np.array(reader).shape == (self.SEEK_HEIGHT, self.SEEK_WIDTH)):
				self.pixd = np.array(reader, dtype=np.float32)
				
			self.pixmin = min(self.pixd.flatten())
			self.pixmax = max(self.pixd.flatten())
			self.variable = 1 / (self.pixmax - self.pixmin)
			self.pixd = (self.pixd - self.pixmin ) * self.variable
			
			im2 = np.uint8(cm.CMRmap(self.pixd)*255)
			#colors the image using matplot lib color map
			o2i = cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)
			o2i = cv2.resize(o2i, (self.width, self.height))
			
			# print(np.array(reader, dtype=np.float32).shape)
			if(o2i.shape == (self.height, self.width, 3)):
				self.current_frame = o2i
			if(np.array(reader, dtype=np.float32).shape == (self.SEEK_HEIGHT, self.SEEK_WIDTH)):
				self.current_data = cv2.resize(np.array(reader, dtype=np.float32), (self.width, self.height))
		except:
			# traceback.print_exc(file=sys.stdout)
			return self.current_frame, self.current_data

		return self.current_frame, self.current_data
