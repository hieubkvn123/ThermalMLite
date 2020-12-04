import os
import cv2
import pickle
import numpy as np
import face_recognition as fr
from face_detection import detect_faces

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as K

graph = tf.get_default_graph()
def get_facenet_model():
    emb_shape = 512
    w_decay = 1e-4
    num_classes = 5000

    # loading facenet
    #model_face = load_model('models/facenet_keras.h5')
    model_face = tf.keras.applications.InceptionV3(include_top=False, input_shape=(170,170,3))

    # freeze all Facenet layers
    for layer in model_face.layers[:]:
        layer.trainable = False

    last = model_face.output
    x = Flatten()(last)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(emb_shape, activation='relu', kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(w_decay))(x)
    x = BatchNormalization()(x)
    x = Dense(num_classes, activation = "softmax", kernel_regularizer=regularizers.l2(w_decay))(x)

    model = Model(inputs=model_face.input, outputs=x, name="model_1")
    model.load_weights('models/facenet.weights.hdf5')
    model = Model(inputs=model.inputs, outputs=model.layers[-3].output)
    model._make_predict_function()
    
    return model

def get_arcface_model():
    emb_shape = 512
    w_decay = 1e-4
    num_classes = 5000

    class ArcFace(Layer):
        def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
            super(ArcFace, self).__init__(**kwargs)
            self.n_classes = n_classes
            self.s = s
            self.m = m
            self.regularizer = regularizers.l2(1e-4/2)#regularizers.get(regularizer)

        def build(self, input_shape):
            super(ArcFace, self).build(input_shape[0])
            self.W = self.add_weight(name='W',
                                    shape=(512, self.n_classes),
                                    initializer='glorot_uniform',
                                    trainable=True,
                                    regularizer=self.regularizer)

        def call(self, inputs):
            x, y = inputs
            c = K.shape(x)[-1]
            # normalize feature
            x = tf.nn.l2_normalize(x, axis=1)
            # normalize weights
            W = tf.nn.l2_normalize(self.W, axis=0)
            # dot product
            original_logits = x @ W
            # add margin
            # clip logits to prevent zero division when backward
            theta = tf.acos(K.clip(original_logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            target_logits = tf.cos(theta + self.m)
            # sin = tf.sqrt(1 - logits**2)
            # cos_m = tf.cos(logits)
            # sin_m = tf.sin(logits)
            # target_logits = logits * cos_m - sin * sin_m
            #
            logits = original_logits * (1 - y) + target_logits * y
            # feature re-scale
            logits *= self.s
            out = tf.nn.softmax(logits)

            return original_logits, out

    # using inceptionv3 as the backbone model
    model_face = tf.keras.applications.InceptionV3(include_top=False, input_shape=(170,170,3))
    labels = Input(shape=(num_classes,))

    # adding custom layers
    last = model_face.output
    x = Flatten()(last)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(emb_shape, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(w_decay))(x)
    x = BatchNormalization()(x)
    logits, x = ArcFace(n_classes=num_classes)([x, labels])

    model = Model(inputs=[model_face.input, labels], outputs=x, name="model_1")
    model.load_weights('models/arcface.weights.hdf5')
    model = Model(inputs=model.inputs[0], outputs=model.layers[-4].output)
    model._make_predict_function()

    return model

model = get_facenet_model()
# model = get_arcface_model()

def preprocessing(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (170, 170))

    ### Standardize the image ###
    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean)/std

    return image


def get_embs_from_folder(folders=None):
    labels = []
    embs = []

    if(folders is None):
        embs = pickle.load(open('validation_encodings.pickle', 'rb'))
        labels = pickle.load(open('known_names.pickle', 'rb'))

        return embs, labels
    
    for folder in folders:
        for (dir_, dirs, files) in os.walk(folder):
            for file_ in files:
                abs_path = dir_ + '/' + file_
                img = cv2.imread(abs_path)

                ### Detect and crop face ###
                box = detect_faces(img)[0]
                x1, y1, x2, y2 = box
                face = img[y1:y2,x1:x2]

                ### Image preprocessing 
                face = preprocessing(face)
                face = np.expand_dims(face, axis=0)
                print('Processing file %s ' % abs_path)

                emb = model.predict(face)[0]
                embs.append(emb)

                label = file_.split('.')[0]
                if(len(file_.split('_')) > 1):
                    label = label.split('_')[0]
                    
                labels.append(label)

    ### Normalize the embeddings ###
    embs = np.array(embs)
    emb_size = np.linalg.norm(embs, axis=1)
    emb_size = emb_size.reshape(-1, 1)
    embs /= emb_size

    labels = np.array(labels)

    pickle.dump(embs, open('validation_encodings.pickle','wb'))
    pickle.dump(labels, open('known_names.pickle', 'wb'))

    return embs, labels

def face_recog(known_faces, frame, model, threshold=0.5):
    ### Detect and crop face ###
    known_embs, known_labels = known_faces
    boxes = detect_faces(frame)
    face_locations = []
    face_id = []

    for box in boxes:
        identity = 'Unknown'
        x1, y1, x2, y2 = box
        face = frame[y1:y2,x1:x2]

        ### check the validity of face's shape ###
        if(np.shape(face) == ()):
            continue 
            
        face = preprocessing(face)

        ### Normalize the new embedding ###
        emb = model.predict(np.array([face]))
        emb /= np.linalg.norm(emb)

        matches = fr.compare_faces(known_embs, emb, tolerance=threshold)
        distances = fr.face_distance(known_embs, emb)
        best_match = np.argmin(distances)
        
        if(matches[best_match]):
            identity = known_labels[best_match]

        print(distances, distances[best_match])

        # print('DISTANCE : ' + str(distances[best_match]))
        face_locations.append((x1,y1,x2,y2))
        face_id.append(identity)

    return face_locations, face_id

### def register_new() ###