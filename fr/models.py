import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

emb_shape = 512
w_decay = 1e-4
num_classes = 93979# 85742 # 5000

class ArcFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.00050, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True)
                                # regularizer=self.regularizer)

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
        # target_logits = tf.cos(theta + self.m)
        sin = tf.sqrt(1 - original_logits**2)
        cos_m = tf.cos(original_logits)
        sin_m = tf.sin(original_logits)
        target_logits = original_logits * cos_m - sin * sin_m

        logits = original_logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)
        # tf.print(x, W, original_logits)

        return out

class AngularMarginPenalty(tf.keras.layers.Layer):
  def __init__(self, n_classes=10, input_dim=512, name='arcface_layer'):
    super(AngularMarginPenalty, self).__init__(name=name)    
    self.s = 10 # the radius of the hypersphere
    self.m1 = 1.0
    self.m2 = 0.45# 5e-8
    self.m3 = 0.15 # 0.0002

    self.n_classes=n_classes
    self.w_init = tf.random_normal_initializer()

    self.W = self.add_weight(name='W',
                                shape=(input_dim, self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=None)
    b_init = tf.zeros_initializer()

    ### For now we are not gonna use bias ###
    
  def set_m1(self, m1):
      self.m1 = m1
  
  def set_m2(self, m2):
      self.m2 = m2

  def set_m3(self, m3):
      self.m3 = m3

  def call(self, inputs):
      x, y = inputs
      c = K.shape(x)[-1]
      ### normalize feature ###
      x = tf.nn.l2_normalize(x, axis=1)

      ### normalize weights ###
      W = tf.nn.l2_normalize(self.W, axis=0)

      ### dot product / cosines of thetas ###
      logits = x @ W
      # tf.print("Original logits : ", logits * (1 - y))
      # tf.print("Original logits : ", tf.reduce_sum(logits * y, axis=1))

      ### add margin ###
      # clip logits to prevent zero division when backward
      theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
      
      marginal_logit = tf.cos(tf.math.maximum(theta*self.m1 + self.m2, 0))  - self.m3 
      logits = logits + (marginal_logit - logits) * y
      # tf.print("Penalized logits : ", tf.reduce_sum(logits * y, axis=1))
      # tf.print("Penalized logits : ", logits * (1-y))

      #logits = logits + tf.cos((theta * self.m)) * y
      # feature re-scale
      logits *= self.s
      out = tf.nn.softmax(logits)

      return out

# using facenet as the backbone model
model_face = tf.keras.applications.InceptionV3(include_top=False, input_shape=(170,170,3))
labels = Input(shape=(num_classes,))
# freeze all layers
# for layer in model_face.layers[:]:
#    layer.trainable = False

# adding custom layers
last = model_face.output
x = Flatten()(last)
x = Dropout(rate=0.5)(x)
x = Dense(emb_shape, name='emb_output')(x)
# x = BatchNormalization()(x) # used for model_3_with_norm
x = AngularMarginPenalty(n_classes=num_classes, input_dim=512, name='arcface_layer')([x, labels])
# x = Dense(num_classes, name='softmax_output', activation='softmax')(x)
# x = ArcFace(n_classes=num_classes, input_dim=512)([x, labels])
facenet = Model(inputs=[model_face.input, labels], outputs=x, name="model_1")

base_path = os.path.dirname(os.path.realpath(__file__))
weights_path = os.path.join(base_path, 'model_94k_faces_glintasia_without_norm.hdf5')
# facenet.load_weights('/home/lattepandademo/ThermalMLite/fr/model_94k_faces_glintasia_without_norm.hdf5')
# facenet._make_predict_function()