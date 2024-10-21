import cv2
import os
import boto3
from flask import Flask
import face_detection
from inference import predict, model_fn, upload_retina_face_mobilenet
import numpy as np
from flask import jsonify

model_dir = '/opt/ml/model'
torch_home = '/tmp'
os.environ['TORCH_HOME'] = torch_home

face_recognizer, label_encoder = model_fn(model_dir)
upload_retina_face_mobilenet(torch_home+'/hub/checkpoints')
detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold = 0.5, nms_iou_threshold = 0.3)

app = Flask(__name__)

# torch_home = 'D:\\UNMSM\\Ciclo X\\Desarrollo de proyecto de tesis II\\Proyecto\\aws ecs for lambda\\facial_recognition'

@app.route("/ping", methods=["GET"])
def ping():
    """
    Healthcheck function.
    """
    return "pong"

@app.route("/invocations", methods=["POST"])
def invocations():
    # Cargar el modelo
    try:
        s3_client = boto3.client('s3')
        url = 'https://7eo8t81vd3.execute-api.us-east-2.amazonaws.com/service-generate-presigned-url'
        print(url)
        # s3=event['Records'][0]['s3']
        # bucket=s3['bucket']['name']
        # key=s3['object']['key']

        bucket='vigilanteye-inference-data'
        key='6/inference-image.jpg'

        key_fragments=key.split('/')
        id=key_fragments[0]
        name=key_fragments[-1].split('.')[0]
        data = {
            'method': 'get_object',
            'id': int(id),
            'name': name,
            'bucket': bucket,
            'key': key
        }

        # response = requests.post(url, json=data)
        # url_presigned = response.json()
        # print(url_presigned)
        # response = requests.get(url_presigned)
        response = s3_client.get_object(Bucket=bucket, Key=key)

        image_data = response['Body'].read()
        image_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        result = predict(face_recognizer, label_encoder, img, detector)
        print(result)
        return jsonify(result)
    
    except Exception as e:
        print(e)
        return jsonify({
            'statusCode': 500,
            'body': str(e)
        })
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)