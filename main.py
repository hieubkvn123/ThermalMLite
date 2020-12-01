from argparse import ArgumentParser
from flask import Flask, render_template, Response
from flask import request
from flask import send_file
from camera_seek import move_file_to_static
from camera_seek import Camera
from camera_seek import get_amb, get_people_per_hour, get_current_incidents, get_people_count, get_current_threshold, get_average_amb

import os
import json
import time
import datetime
import threading
import camera_seek as camera
import requests
import logging
import numpy as np

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'HieuDepTry'

### This is used to disable logging into to console ###
parser = ArgumentParser()
parser.add_argument("-v", "--verbose", required=False, help='Verbose mode')
parser.add_argument("-c", "--case", required=False, help="Case orientation (horizontal or vertical)")
args = vars(parser.parse_args())

orientation = 'vertical' # vertical by default
if(args['case']):
	orientation = args['case']

### Config the case orientation ###
camera.set_orientation(orientation)

### Check if incidents dir is present ###
if(not os.path.exists("static/img/incidents")):
	print("[INFO] Incident directory does not exist, creating ...")
	os.mkdir("static/img/incidents")

### Debugger mode on ###
if(not args['verbose']):
	log = logging.getLogger("werkzeug")
	log.setLevel(logging.ERROR)

SEEK_THERMAL = True
TERABEE = False
types = ['SEEK_THERMAL', 'TERABEE']
mask = [SEEK_THERMAL, TERABEE]

@app.route('/')
def index():
	return render_template('index.html', camera_type=np.array(types)[mask])

def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# cam = VideoCamera()
@app.route('/video_feed')
def video_feed():
	return Response(gen(Camera()),
					mimetype='multipart/x-mixed-replace; boundary=frame')


# In order to get the data needed from the thermal camera
# we need to create a get function that pulls ambient temp and people per hour data
# from the camera module (independently from the class Camera)
# request one and get both data
@app.route("/get_data", methods = ['POST'])
def get_data():
	if(request.method == 'POST'):
		amb_data, amb_x_axis = get_amb()
		pph_data, pph_x_axis = get_people_per_hour()

		amb_data = list(np.array(amb_data, dtype=str))
		pph_data = list(np.array(pph_data, dtype=str))

		data = [amb_data, pph_data]
		axis = [amb_x_axis, pph_x_axis]

		plot_data = [data, axis]

		json_data = json.dumps(plot_data)
		return json_data

def get_incidents_offline():
	INCIDENT_DIR = 'static/img/incidents'
	files = os.listdir(INCIDENT_DIR)

	return files

@app.route("/get_incidents", methods = ['POST'])
def get_incidents():
	if(request.method == 'POST'):
		# INCIDENT_DIR = 'static/img/incidents'

		files = get_incidents_offline() # os.listdir(INCIDENT_DIR)
		json_ = json.dumps(files)

		return json_

@app.route("/change_camera_view", methods = ['POST'])
def change_camera_view():
	if request.method == 'POST':
		camera.set_camera_view(request.form['camera_view'])

		return 'success'

@app.route("/download", methods = ['POST','GET'])
def download():
	file_name = move_file_to_static()	

	return send_file('static/' + file_name, as_attachment=True)

@app.route("/poll_incident", methods = ['POST'])
def poll_incident():
	# client polls for top four incidents
	current_incidents = get_current_incidents()
	incident_count = get_people_count()
	average_amb = get_average_amb()

	abs_file_name = []
	for file_name in current_incidents:
		abs_file_name.append("/static/img/incidents/" + file_name)

	# abs_file_name_json = json.dumps(abs_file_name)

	objects = {}
	objects['incident_img'] = abs_file_name
	objects['incident_count'] = incident_count
	objects['average_amb'] = average_amb

	objects_json = json.dumps(objects)

	return objects_json

@app.route("/get_thresh", methods=['POST'])
def get_thresh():
	current_thresh = get_current_threshold()
	print(current_thresh)

	return str(current_thresh)

@app.route("/set_thresh", methods = ['POST'])
def set_thresh():
	threshold = float(request.form['thresh'])
	camera.set_temp_threshold(threshold)

	return 'success'

#@app.route("/get_instore", methods=['POST'])
#def get_count():
#	if(request.method == 'POST'):
#		count = requests.post("https://www.compasswool.com/flask/count",data={"type":"receiver"}).text
#
#		return count

@app.route("/set_language", methods = ["POST"])
def set_language():
	if(request.method == 'POST'):
		language = request.form['language']
		camera.set_current_language(language)

		return 'success'

@app.route("/set_metric", methods = ["POST"])
def set_metric():
	if(request.method == 'POST'):
		metric = request.form['metric']
		camera.set_current_metric(metric)

		return 'success' 

@app.route("/shutdown", methods=['POST'])
def shutdown():
	if(request.method == 'POST'):
		shutdown = threading.Thread(target=os.system, args=("sudo sh shutdown.sh",))		
		shutdown.start() 

		print("[INFO] Server is restarting ...")

		return 'success'

@app.route("/reboot", methods = ["POST"])
def reboot():
	if(request.method == 'POST'):
		reboot = threading.Thread(target = os.system, args = ("sudo sh reboot.sh",))
		reboot.start()

		print("[INFO] System rebooting ... ")

		return 'success'

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=False, port=8000)

# Issues : The app is not recording data correctly
# there are thousands of partitions in the data log and they are emptys
