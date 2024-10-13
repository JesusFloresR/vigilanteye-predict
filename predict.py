import cv2
import numpy as np
import imutils
import os
from joblib import load
import json

detector = None

def initModule():
  global initialized
  global detector

  if initialized:
     return
  import face_detection
  detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold = 0.5, nms_iou_threshold = 0.3)
  initialized = True

def extract_face(img):
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

    if prediction[1] < umbral_confianza:
        clase_predicha = np.argmax(prediction)
        print("clase_predicha: ", clase_predicha)
        etiqueta_predicha = label_encoder.inverse_transform([prediction[0]])[0]
    else: 
        etiqueta_predicha = "Desconocido"
    
    return etiqueta_predicha

def model_fn(model_dir):
    # Cargar el modelo
    initModule()
    print("Cargando modelo...")
    model_path = os.path.join(model_dir, 'modeloLBPHFace.xml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model_path)
    print("Modelo cargado")

    print("Cargando Label Encoder...")
    model_path = os.path.join(model_dir, 'label_encoder.pkl')
    label_encoder = load(model_path)
    print("Label Encoder cargado")
    return face_recognizer, label_encoder

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/x-image':
        nparr = np.frombuffer(request_body, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return img
    raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    face_recognizer, label_encoder = model
    faces = extract_face(input_data)  # Define esta función según tu lógica
    predictions = []
    umbral=65
    if faces is not None:
        for face, xmin, ymin, xmax, ymax in faces:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            prediction = face_recognizer.predict(gray)
            predictions.append(label)  # O cualquier otro formato que necesites
            label = get_label(prediction, label_encoder, umbral)
            predictions.append([label, xmin, ymin, xmax, ymax])

    return predictions

def output_fn(predictions, content_type):
    return json.dumps(predictions)