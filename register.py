import os
import cv2
import face_recognition
from imutils.video import WebcamVideoStream

print("Enter your name : ", end = "")

name = input()

vs = WebcamVideoStream(src = 0).start()

def create_masks(path, mask_types, output_dir):
	cmd = "cd MaskTheFace && " + \
		  "python3 mask_the_face.py --path {} --mask_type '{}' --output_dir {}"

	for mask_type in mask_types:
		os.system(cmd.format(path, mask_type, output_dir))

while(True):
	frame = vs.read()
	frame = cv2.flip(frame, flipCode=-1)
	frame_ = frame.copy()

	locations = face_recognition.face_locations(frame_)

	for (top, right, bottom, left) in locations:
		cv2.rectangle(frame_, (left, top), (right, bottom), (0,255,0), 2)

	key = cv2.waitKey(1)
	if(key == ord("q")):
		break
	elif(key == ord("s")):
		write_path = os.path.join('faces', name + '.jpg')
		mask_the_face_path = '../' + write_path
		
		cv2.imwrite(write_path, frame)
		cv2.putText(frame, "Image saved" ,(10,10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 2)
		create_masks(mask_the_face_path, ['surgical', 'cloth'], '../masked')

	cv2.imshow("Frame", frame_)

vs.stop()
cv2.destroyAllWindows()
