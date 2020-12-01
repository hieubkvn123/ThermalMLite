import cv2
import face_recognition
from imutils.video import WebcamVideoStream

print("Enter your name : ", end = "")

name = input()

vs = WebcamVideoStream(src = 0).start()

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
		cv2.imwrite("faces/" + name + ".jpg", frame)
		cv2.putText(frame, "Image saved" ,(10,10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 2)

	cv2.imshow("Frame", frame_)

vs.stop()
cv2.destroyAllWindows()
