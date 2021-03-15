import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import Model
from models import facenet

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--export-path', required=True, help="Model export path")
args = vars(parser.parse_args())

# Constants
weights = 'model_94k_faces_glintasia_without_norm.hdf5'
hdf5    = 'model_94k_faces_glintasia_without_norm.h5'
name    = 'arcface_keras'
path    = args['export_path']

# Load model with weights file
print('[INFO] Loading model weights ...')
facenet.load_weights(weights)

# Truncate model
print('[INFO] Truncating model...')
model = Model(inputs=facenet.inputs[0], outputs=facenet.get_layer('emb_output').output)

# Export model to hdf5 format first
print('[INFO] Saving model to hdf5...')
model.save(hdf5)

# Export model to tf SavedModel format
print('[INFO] Saving model to tf SavedModel format ... ')
model_path = os.path.join(path, name)
model_path = os.path.join(model_path, "1")
models.save_model(model, model_path, save_format="tf")
