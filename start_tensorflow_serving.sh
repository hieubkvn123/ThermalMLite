sudo docker run -t --rm -p 8502:8501 -v "/home/lattepandademo/arcface_keras:/models/arcface_keras" -e "MODEL_NAME=arcface_keras" tensorflow/serving
