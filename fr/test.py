import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from models import facenet
from sklearn.decomposition import PCA

base_path = os.path.dirname(os.path.realpath(__file__))
facenet.load_weights(os.path.join(base_path, 'model_94k_faces_glintasia_without_norm_.hdf5'))
facenet = tf.keras.models.Model(inputs=facenet.inputs[0], outputs=facenet.get_layer('emb_output').output)

def neutralize_image(img):
    r,g,b = cv2.split(img)
    r = cv2.equalizeHist(r)
    g = cv2.equalizeHist(g)
    b = cv2.equalizeHist(b)

    rgb = cv2.merge((r,g,b))
    return rgb

def preprocessing(img):
    img = cv2.resize(img, (170, 170))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = neutralize_image(img)

    img = (img - 127.5) / 127.5

    return img

images = []
labels = []
datadir = os.path.join(base_path, 'data')
for (dir_, dirs, files) in os.walk(datadir):
    if(dir_ != datadir):
        label = dir_.split('/')[-1]
        for file_ in files:
            if(file_.endswith('.jpg')):
                abs_path = os.path.join(dir_, file_)
                print('Preprocessing file %s' % abs_path)
                img = cv2.imread(abs_path)
                img = preprocessing(img)

                images.append(img)
                labels.append(label)


images = np.array(images)
labels = np.array(labels)
assert(images.shape[0] == labels.shape[0])

embeddings = facenet.predict(images)
embeddings /= np.linalg.norm(embeddings, axis=1).reshape(-1, 1)
embeddings_pca = PCA(n_components = 3).fit_transform(embeddings)
embeddings_pca_norm = embeddings_pca / np.linalg.norm(embeddings_pca, axis=1).reshape(-1, 1)

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

for label in np.unique(labels):
    cluster = embeddings_pca_norm[labels == label]
    ax1.scatter(cluster[:,0], cluster[:,1], cluster[:,2])

    cluster = embeddings_pca[labels == label]
    ax2.scatter(cluster[:,0], cluster[:,1], cluster[:,2])

ax1.set_title('Face embeddings (normalized)')
ax2.set_title('Face embeddings (not normalized)')
plt.show()
