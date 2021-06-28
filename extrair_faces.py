from mtcnn import MTCNN
from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray

detector= MTCNN()

# Carrega as imagens do path, detecta e extrai as faces
def extrair_face(arquivo, size =(160,160)):

    img = Image.open(arquivo) #caminho do path

    img = img.convert('RGB') # converter em RGB

    array = asarray(img)

    results = detector.detect_faces(array)

    x1, y1, width, height = results[0] ['box']

    x2, y2 = x1 + width, y1 + height

    face = array [y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(size=(160,160))

    return image

def flip_image(image): #flipa a imagem
    img = image.transpose(Image.FLIP_LEFT_RIGHT)
    return img
#carrega  as fotos do diretorio
def load_fotos(directory_src, directory_target):

    for filename in listdir(directory_src):

        path = directory_src + filename
        path_tg = directory_target + filename
        path_tg_flip = directory_target + "flip-"+filename #espelhar imagem para melhor tratamento

        try:
            face = extrair_face(path)
            flip = flip_image(face)

            face.save(path_tg, "JPEG", quality=100, optimize=True, progressive=True)
            flip.save(path_tg_flip, "JPEG", quality=100, optimize=True, progressive=True)

        except:
            print("Erro, verifique se tem pasta correspondente e o formato a imagem {}".format(path))
#recebe o diretorio das fotos e das faces
def load_dir(directory_src, directory_target):
    
    for subdir in listdir(directory_src):

        path = directory_src + subdir + "\\"
        path_tg= directory_target + subdir + "\\"

        if not isdir(path):
            continue

        load_fotos(path, path_tg)

# Diretorio que onde vai importar e exportar as imagens, primeira linha para importação e segunda para exportação
if __name__ == '__main__':

    load_dir("\\\\172.20.9.172\\rostos\\fotos\\",
    "\\\\172.20.9.172\\rostos\\faces\\")
    
    '''load_dir("C:\\Users\\User\\Documents\\reconhecimento facial\\reconhecimento_e_detectacao\\rostos\\fotos\\",
    "C:\\Users\\User\\Documents\\reconhecimento facial\\reconhecimento_e_detectacao\\rostos\\faces\\")
    '''