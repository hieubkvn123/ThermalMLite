import os
import cv2
import pickle
import numpy as np
import face_recognition as fr
from face_detection import detect_faces

import tensorflow as tf
from scipy.spatial.distance import cdist
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

def get_arcface_mobilenet_model():
    input_shape=(112,112,3)
    emb_shape=256
    bn_momentum=0.99
    bn_epsilon=0.001

    mobilenet = tf.keras.applications.MobileNet(input_shape=input_shape, include_top=False, weights="imagenet")
    inputs = mobilenet.inputs[0]
    output = mobilenet.outputs[0]

    nn = tf.keras.layers.Conv2D(512, 1, use_bias=False, padding="same")(output)
    nn = tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(nn)
    nn = tf.keras.layers.PReLU(shared_axes=[1, 2])(nn)

    nn = tf.keras.layers.DepthwiseConv2D(int(nn.shape[1]), depth_multiplier=1, use_bias=False)(nn)
    nn = tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(nn)
    nn = tf.keras.layers.Conv2D(emb_shape, 1, use_bias=False, activation=None, kernel_initializer="glorot_normal")(nn)
    nn = tf.keras.layers.Flatten()(nn)

    embedding = tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name="embedding", scale=True)(nn)
    basic_model = tf.keras.models.Model(inputs, embedding, name='ArcFace_MobilenetBackBone') 
    basic_model.load_weights("models/mobilenet_arcface.weights.hdf5")

    return basic_model

def get_arcface_model():
    emb_shape = 512
    w_decay = 1e-4
    num_classes = 5000

    class AngularMarginPenalty(tf.keras.layers.Layer):
      def __init__(self, n_classes=10, input_dim=512):
        super(AngularMarginPenalty, self).__init__()    
        self.s = 30 # the radius of the hypersphere
        self.m1 = 1.0
        self.m2 = 0.003
        self.m3 = 0.02
        self.n_classes=n_classes
        self.w_init = tf.random_normal_initializer()

        self.W = self.add_weight(name='W',
                                    shape=(input_dim, self.n_classes),
                                    initializer='glorot_uniform',
                                    trainable=True,
                                    regularizer=None)
        b_init = tf.zeros_initializer()

        ### For now we are not gonna use bias ###
        
      def call(self, inputs):
          x, y = inputs
          c = K.shape(x)[-1]
          ### normalize feature ###
          x = tf.nn.l2_normalize(x, axis=1)

          ### normalize weights ###
          W = tf.nn.l2_normalize(self.W, axis=0)

          ### dot product / cosines of thetas ###
          logits = x @ W

          ### add margin ###
          ''' 
            in the paper we have theta + m but here I am just gonna decrease theta 
            this is because most theta are within [0,pi/2] - in the decreasing region of cosine func
          '''
          
          # clip logits to prevent zero division when backward
          theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
          
          marginal_logit = tf.cos((tf.math.maximum(theta*self.m1 + self.m2, 0))) # - self.m3 
          logits = logits + (marginal_logit - logits) * y
          
          #logits = logits + tf.cos((theta * self.m)) * y
          # feature re-scale
          logits *= self.s
          out = tf.nn.softmax(logits)

          return out

    # using inceptionv3 as the backbone model
    model_face = tf.keras.applications.InceptionV3(include_top=False, input_shape=(170,170,3))
    labels = Input(shape=(num_classes,))

    # adding custom layers
    last = model_face.output
    x = Flatten()(last)
    #x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(emb_shape)(x)#, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(w_decay))(x)
    # x = BatchNormalization()(x)
    # logits, x = ArcFace(n_classes=num_classes)([x, labels])
    x = AngularMarginPenalty(n_classes=num_classes, input_dim=512)([x, labels])

    model = Model(inputs=[model_face.input, labels], outputs=x, name="model_1")
    model.load_weights('models/arcface.weights.hdf5')
    model = Model(inputs=model.inputs[0], outputs=model.layers[-3].output)
    model._make_predict_function()

    return model

def get_arcface_model_1():
    emb_shape = 512
    w_decay = 1e-4
    num_classes = 10000

    class AngularMarginPenalty(tf.keras.layers.Layer):
      def __init__(self, n_classes=10, input_dim=512):
        super(AngularMarginPenalty, self).__init__()    
        self.s = 30 # the radius of the hypersphere
        self.m1 = 1.0
        self.m2 = 0.003
        self.m3 = 0.02
        self.n_classes=n_classes
        self.w_init = tf.random_normal_initializer()

        self.W = self.add_weight(name='W',
                                    shape=(input_dim, self.n_classes),
                                    initializer='glorot_uniform',
                                    trainable=True,
                                    regularizer=None)
        b_init = tf.zeros_initializer()

        ### For now we are not gonna use bias ###
        
      def call(self, inputs):
          x, y = inputs
          c = K.shape(x)[-1]
          ### normalize feature ###
          x = tf.nn.l2_normalize(x, axis=1)

          ### normalize weights ###
          W = tf.nn.l2_normalize(self.W, axis=0)

          ### dot product / cosines of thetas ###
          logits = x @ W

          ### add margin ###
          ''' 
            in the paper we have theta + m but here I am just gonna decrease theta 
            this is because most theta are within [0,pi/2] - in the decreasing region of cosine func
          '''
          
          # clip logits to prevent zero division when backward
          theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
          
          marginal_logit = tf.cos((tf.math.maximum(theta*self.m1 + self.m2, 0))) # - self.m3 
          logits = logits + (marginal_logit - logits) * y
          
          #logits = logits + tf.cos((theta * self.m)) * y
          # feature re-scale
          logits *= self.s
          out = tf.nn.softmax(logits)

          return out

    # using inceptionv3 as the backbone model
    model_face = tf.keras.applications.InceptionV3(include_top=False, input_shape=(170,170,3))
    labels = Input(shape=(num_classes,))

    # adding custom layers
    last = model_face.output
    x = Flatten()(last)
    #x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(emb_shape, name='emb_output')(x)#, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(w_decay))(x)
    # x = BatchNormalization()(x)
    # logits, x = ArcFace(n_classes=num_classes)([x, labels])
    x = AngularMarginPenalty(n_classes=num_classes, input_dim=512)([x, labels])
    # x = Dense(num_classes, name='softmax_output', activation='softmax')(x)

    model = Model(inputs=[model_face.input, labels], outputs=x, name="model_1")
    model.load_weights('models/arcface_2.weights.hdf5')
    #model.load_weights('models/model_3_with_norm.hdf5')
    model = Model(inputs=model.inputs[0], outputs=model.get_layer('emb_output').output)
    model._make_predict_function()

    return model

# model = get_facenet_model()
# model = get_arcface_model()
# model = get_arcface_model_1()
model = get_arcface_mobilenet_model()

def preprocessing(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (170, 170))

    ### Standardize the image ###
    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean)/std

    return image

def preprocessing_1(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (170, 170))
    image = cv2.resize(image, (112, 112))
    image = (image - 127.5) * 0.0078125

    ### Standardize the image ###
    # mean = np.mean(image)
    # std = np.std(image)
    # image = (image - mean)/std

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
                face = preprocessing_1(face)
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
            
        face = preprocessing_1(face)

        ### Normalize the new embedding ###
        emb = model.predict(np.array([face]))
        emb /= np.linalg.norm(emb)

        matches = fr.compare_faces(known_embs, emb, tolerance=threshold)
        distances = fr.face_distance(known_embs, emb)
        best_match = np.argmin(distances)
        
        if(matches[best_match]):
            identity = known_labels[best_match]

        # print(distances, distances[best_match])

        # print('DISTANCE : ' + str(distances[best_match]))
        face_locations.append((x1,y1,x2,y2))
        face_id.append(identity)

    return face_locations, face_id

def face_recog_adaptive(known_faces, frame, model, thresholds):
    ### Detect and crop face ###
    margin = 0.0

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
            
        face = preprocessing_1(face)

        ### Normalize the new embedding ###
        emb = model.predict(np.array([face]))
        emb /= np.linalg.norm(emb)

        # print(emb.shape)
        dist_mat = 1 - cdist(np.array(emb), known_embs, 'cosine')
        dist_mat = dist_mat[0]
        best_match = np.argmax(dist_mat)

        print(dist_mat[best_match], thresholds[best_match])
        if(dist_mat[best_match] >= thresholds[best_match] + margin):
            identity = known_labels[best_match]

        face_locations.append((x1,y1,x2,y2))
        face_id.append(identity)

    return face_locations, face_id

def get_threshold(embs, labels, distance='cosine'):
    dist_matrix = cdist(embs, embs, distance)
    sigmas = []

    for i, label in enumerate(labels):
        dist = dist_matrix[i]
        dist = dist[np.where(labels != label)]
        dist = 1 - dist 
        sigma = dist[np.argmax(dist)]
        sigmas.append(sigma)

    return sigmas

### def register_new() ###