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

cv2.namedWindow("test")
#empty array as a buffer for the data
pixd = np.zeros([150,200],dtype=float)
#min and max value so less computation on finding min max per frame
pixmax = 40
pixmin = 20
variable = 1/(pixmax - pixmin)

print("[INFO] Running thermal app ... ")
t_thermal = multiprocessing.Process(target=os.system, args=("./seekware-simple",))
t_thermal.start()

while True:
	line_ = []
	try:
		start = time.time()
		thermograph = open("thermography.csv",'r')
		reader = list(csv.reader(thermograph))
		
		for x, line in enumerate(reader):
			line_ = line
			for i in range(len(line)):
				if(line[i] == ''):
					continue
				pixd[x,i] = (float(line[i]) - pixmin) * variable

		im2 = np.uint8(cm.CMRmap(pixd)*255)
		#colors the image using matplot lib color map
		o2i = cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)
		o2i = cv2.resize(o2i, (800, 600))
		cv2.imshow("test", o2i)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
			## print("fps:", 1/(time.time() - start))
	except:
		print(line_)
		traceback.print_exc(file=sys.stdout)
		continue

print("[INFO] Stopping thermal ... ")
os.system("killall seekware-simple")