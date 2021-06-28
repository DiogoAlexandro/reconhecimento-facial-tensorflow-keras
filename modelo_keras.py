import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
import keras
df_desconhecidos = pd.read_csv("faces_desconhecidos.csv") # ler o arquivo csv dos desconhecidos, lembrando que essa é uma base de dados com varias pessoas desconhecidas
df_conhecidos = pd.read_csv("faces.csv" ) # base das pessoas

df =  (pd.concat([df_desconhecidos, df_conhecidos])) # concatena o faces com o faces desconhecidos. 
df = df.astype({'target':str})#obs: com numeros de cpf usar o astype({'target':str}) para converter o target para string
X = np.array(df.drop("target", axis=1))

y = np.array(df.target)

#Mistura tudo
from sklearn.utils import shuffle

X, y = shuffle(X, y, random_state=0)

#modelo de teste e validacao
from sklearn.model_selection import train_test_split

trainX, valX, trainY, valY = train_test_split(X, y, test_size=0.20, random_state=42)

#normalizando
from sklearn.preprocessing import Normalizer

norm = Normalizer(norm="l2")
trainX = norm.transform(trainX)
valX = norm.transform(valX)
# Tratar as labels para algoritimo de inteligencia artificial binarizando

from sklearn.preprocessing import LabelEncoder

np.unique(trainY)
classes = len(np.unique(trainY))
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)
np.unique(trainY)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(valY)
valY = out_encoder.transform(valY)
np.unique(valY)

                    ###Keras

from tensorflow.keras.utils import to_categorical
trainY = to_categorical(trainY)
valY = to_categorical(valY)

from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(128, activation="relu", input_shape=(128,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(classes, activation="softmax"))

model.compile(optimizer="adam",
loss='categorical_crossentropy',
metrics=['accuracy'])

batch_size = 50 #  variavel
epochs = 40 # variavel

history = model.fit(trainX, trainY,
                    epochs=epochs,
                    validation_data = (valX,valY),
                    batch_size=batch_size)
# sumariza historico de acuracia
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# sumariza historico de loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

yhat_val = model.predict(valX)

valY = np.argmax(valY,axis = 1)
yhat_val = np.argmax(yhat_val,axis = 1)

                                ### Matriz confusão

from sklearn.metrics import confusion_matrix
# Função para mostrar a  matriz confusão
def print_confusion_matrix(model_name, valY, yhat_val):
    cm = confusion_matrix(valY, yhat_val)
    #total = sum(sum(cm))
    acc = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]) #acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    print("Modelo: {}".format(model_name))
    print("Acurácia: " + str(acc*100)+ "%") #print("Acurácia: {:.4f}".format(acc))
    print("Sensitividade: " + str(sensitivity*100)+ "%") #print("Sensitividade: {:.4f}".format(sensitivity))
    print("Especificidade: " + str(specificity*100)+ "%") #print("Especificidade: {:.4f}".format(specificity))

    from mlxtend.plotting import plot_confusion_matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm ,  figsize=(5, 5))
    plt.show()

print_confusion_matrix("Keras", valY, yhat_val)
model.save('modelo_faces.h5') # salva o modelo em h5