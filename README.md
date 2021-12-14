# ThermalXAndFR
# Installation for Vertical case:
```bash
$ sudo sh install.sh vertical 
```
# Installation for Horizontal case :	
```bash 
$ sudo sh install.sh horizontal 
```

# Preparing embedders :
	1. Keras to tf SavedModel
```python
from tensorflow.keras import models
from tensorflow.keras.models import Model
from fr.models import facenet

# Loading weights
facenet.load_weights("model_94k_faces_glintasia_without_norm.hdf5")

# Truncate the model
model = Model(inputs=facenet.inputs[0], outputs=facenet.get_layer("emb_output").output)
model.save("model_94k_faces_glintasia_without_norm.h5") # store to hdf5 format

# Save model in tf SavedModel format
models.save_model(model, "<abs_path>/<model_name>/<version_name>")

'''
	Now inside <abs_path>/<model_name>/<version_name> should have:
		1. assets folder
		2. variables folder
		3. saved_model.pb file
'''
```

# Starting tensorflow serving :
```bash
$ sudo sh start_tensorflow_serving.sh 
```
	This script and run_main.sh script must be run concurrently

##
