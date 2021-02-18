import os
import cv2
import time
import numpy as np
import tensorflow as tf

from models import facenet
from detect_utils import detect_and_align

base_path = os.path.dirname(os.path.realpath(__file__))
weights_path = os.path.join(base_path, 'model_94k_faces_glintasia_without_norm.hdf5')
facenet.load_weights(weights_path)
facenet = tf.keras.models.Model(inputs=facenet.inputs[0], outputs=facenet.get_layer('emb_output').output)

registration_folder = os.path.join(base_path, 'identities')
if(not os.path.exists(registration_folder)):
    print('[INFO] Making registration folder ...')
    os.mkdir(registration_folder)

print('What is your name : ', end='')
name = input()

if(not os.path.exists(os.path.join(registration_folder, name))):
    print('[INFO] Making identity folder for id %s' % name)
    os.mkdir(os.path.join(registration_folder, name))
    os.mkdir(os.path.join(registration_folder, name, 'imgs'))

camera = cv2.VideoCapture(0)
start = time.time()
frame_count = 0
NUM_SECS = 5
images = []

def preprocessing(img):
    img = cv2.resize(img, (170, 170))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img - 127.5) / 127.5

    return img

while(True):
    ret, frame = camera.read()
    now = time.time()
    output_path = os.path.join(registration_folder, name, 'imgs', '%s_%d.jpg' % (name, frame_count))
    
    if(now - start >= NUM_SECS):
        break

    if(ret == True):
        faces, locations = detect_and_align(frame)
        if(len(faces) > 0):
            cv2.imwrite(output_path, frame)
            
            face = faces[0]
            location = locations[0]
            images.append(preprocessing(face))

            x1, y1, x2, y2 = location 
            frame = cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 2)

            cv2.imshow('Face', face)

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if(key == ord('q')):
            break

    frame_count += 1

camera.release()
cv2.destroyAllWindows()

print('[INFO] Generating embeddings ... ')
embeddings = facenet.predict(np.array(images))
embeddings = embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1)

with open(os.path.join(registration_folder, name, '%s.npy' % name), 'wb') as f:
    np.save(f, embeddings)
    print('[INFO] Saved normalized embeddings to %s' % os.path.join(registration_folder, name, '%s.npy' % name))

