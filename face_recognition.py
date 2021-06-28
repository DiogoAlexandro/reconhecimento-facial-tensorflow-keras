from mtcnn.mtcnn import MTCNN
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2
from sklearn.preprocessing import Normalizer
import os

path = "\\\\172.20.9.172\\rostos\\faces\\"  #diretorio das faces
# Pega  o  nome  das pastas de cada  pessoa
nomes = os.listdir(path)
pessoa = nomes
num_classes = len(pessoa)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

detector = MTCNN()
facenet = load_model("facenet_keras.h5", compile=False)
model = load_model("modelo_faces.h5")

# Função para extrair as faces das fotos
def extract_face(image, box, required_size=(160, 160)):
    pixels = np.asarray(image)
    
    x1, y1, width, height = box
    
    x2, y2 = x1 + width, y1 + height
    
    face = pixels[y1:y2, x1:x2]
    
    image = Image.fromarray(face)
    image = image.resize(required_size)
    
    return np.asarray(image)

# Função que gera os embeddings
def get_embedding(facenet, face_pixels):
    face_pixels = face_pixels.astype('float32')
    
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean)/ std
    
    samples = np.expand_dims(face_pixels, axis=0)
    
    yhat = facenet.predict(samples)
    return yhat[0]

# Laço que verifica em  tempo real as faces, caso detecte mostra o nome da pessoa
while True:
    _, frame= cap.read()
    
    faces = detector.detect_faces(frame)
    
    for face in faces:
        confidence = face['confidence'] * 100
        
        if confidence >= 98:
            try:
                x1, y1, w, h = face['box']
                face = extract_face(frame, face['box'])
                face = face.astype("float32")/255
                
                emb = get_embedding(facenet, face)
                
                tensor = np.expand_dims(emb, axis=0)
                norm = Normalizer(norm="l2")
                tensor = norm.transform(tensor)
                
                classe = np.argmax(model.predict(tensor)[0]) #model.predict_classes(tensor)[0] #deprecated
                prob = model.predict(tensor) #model.predict_proba(tensor) #deprecated
                prob = prob[0][classe] * 100
            except ValueError:
                pass
            if prob>=98:
            #COR EM (BGR) dos retangulos em  volta, if para desconhecido else para conhecidos, verificar qual a posição dos desconhecidos.
                if classe==(num_classes-1):
                    color = (0,0,255)
                else:
                    color = (192, 255, 119)
                    
                user = str(pessoa[classe]).upper()
                
                cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), color, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(frame, user, (x1, y1-10), font, font_scale, color, 
                            thickness=1)
            
            
    cv2.imshow("Face Recognittion", frame) # Mostra o video na tela
    
    key = cv2.waitKey(1)
    
    if key==27: #ESC
        break

cap.release()
cv2.destroyAllWindows()
