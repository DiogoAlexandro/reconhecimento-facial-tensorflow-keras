from operator import ge
from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray, expand_dims
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def load_face(filename):
    #carregando imagem de arquivo
    image = Image.open(filename)

    #converter em RGB
    image = image.convert("RGB")

    return asarray(image)
#carrega as faces
def load_faces(directory_src):

    faces = list()

    #iterando arquivos
    for filename in listdir(directory_src):

        path = directory_src + filename

        try:
            faces.append(load_face(path))
        except:
            print("Erro na imagem {}".format(path))
    return faces

def load_fotos(directory_src):
    X, y = list(), list()
    # iterar pastas por classes
    for subdir in listdir(directory_src):
        #path
        path = directory_src + subdir + '\\'

        if not isdir(path):
            continue

        faces = load_faces(path)

        labels = [subdir for _ in range(len(faces))]
        #sumarizar progresso
        print('>Carregadas %d faces da classe: %s' % (len(faces), subdir))

        X.extend(faces)
        y.extend(labels)

    return asarray(X), asarray(y)

# Carregando todas as imagens das faces
trainX, trainy = load_fotos(directory_src="\\\\172.20.9.172\\rostos\\faces\\")
print(trainX.shape)
print(trainy.shape)
#importando modelo keras
model = load_model('facenet_keras.h5') # ,compile=False

# extração de caracteristicas (padronização)
def get_embedding(model, face_pixels):

    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean)/std

    #Transforma a face em 1 único exemplo

    samples = expand_dims(face_pixels, axis=0)

    yhat = model.predict(samples)

    return yhat[0]

newTrainX = list()
#transformação das faces para dataframe
for face in trainX:
    embedding = get_embedding(model, face)
    newTrainX.append(embedding)

newTrainX = asarray(newTrainX)
newTrainX.shape

df = pd.DataFrame(data=newTrainX)
df['target'] = trainy

df.to_csv('faces.csv', index=False) # exportação do arquivo para csv