import imutils
import cv2
import os
import requests
import boto3
import tarfile
from joblib import load
import numpy as np

def upload_retina_face_mobilenet(destination_directory):
    id = '6'
    name = ''
    bucket = 'vigilanteye-models'
    # bucket='vigilenteye-faces-video'
    key='RetinaFace_mobilenet025.pth'
    data = {
        'method': 'get_object',
        'id': int(id),
        'name': name,
        'bucket': bucket,
        'key': key
    }

    url = 'https://7eo8t81vd3.execute-api.us-east-2.amazonaws.com/service-generate-presigned-url'

    response = requests.post(url, json=data)
    url_presigned = response.json()
    print(url_presigned)

    response = requests.get(url_presigned)
    if response.status_code == 200:
        # Asegúrate de que el directorio /tmp existe
        print('creando')
        os.makedirs(destination_directory, exist_ok=True)
        print('creo')

        # Guarda el contenido en un archivo local
        with open(destination_directory + '/RetinaFace_mobilenet025.pth', 'wb') as f:
            f.write(response.content)
        print(f'Archivo descargado y guardado en {destination_directory}')
    else:
        print('Error al descargar el archivo:', response.status_code)

def model_fn(model_dir):
    # Carga el modelo LBPH
    model_path = os.path.join(model_dir, 'modeloLBPHFace.xml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model_path)

    # Carga el label encoder
    model_path = os.path.join(model_dir, 'label_encoder.pkl')
    label_encoder = load(model_path)

    return face_recognizer, label_encoder

def extract_face(img, detector):
    frame = imutils.resize(img, width=640)
    auxFrame = frame.copy()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img)
    detections = detector.detect(frame)
    faces = []
    face = None
    xmin_ = None
    ymin_ = None
    xmax_ = None
    ymax_ = None

    if len(detections)==0:
        return face, xmin_, ymin_, xmax_, ymax_
    
    for xmin,ymin,xmax,ymax,precision in detections:
        xmin_, ymin_, xmax_, ymax_ = xmin, ymin, xmax, ymax
        # print(ymin,ymax,xmin,xmax)
        face = auxFrame[int(ymin):int(ymax),int(xmin):int(xmax)]
        face = cv2.resize(face,(150,150),interpolation=cv2.INTER_CUBIC)
        faces.append([face, xmin_, ymin_, xmax_, ymax_])
    
    return faces

def get_label(prediction, label_encoder, umbral_confianza):
    etiqueta_predicha = None
    metric = prediction[1]
    if metric < umbral_confianza:
        clase_predicha = np.argmax(prediction)
        print("clase_predicha: ", clase_predicha)
        etiqueta_predicha = label_encoder.inverse_transform([prediction[0]])[0]
    else: 
        etiqueta_predicha = "Desconocido"
    
    return etiqueta_predicha, metric

def predict(face_recognizer, label_encoder, img, detector):
    faces = extract_face(img, detector)
    predictions = []
    umbral=70
    if faces is not None:
        for face, xmin, ymin, xmax, ymax in faces:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            prediction = face_recognizer.predict(gray)
            label, metric = get_label(prediction, label_encoder, umbral)
            # predictions.append(label)  # O cualquier otro formato que necesites
            predictions.append([label, float(xmin), float(ymin), float(xmax), float(ymax), metric])

    return predictions