from flask import Flask, request
from ultralytics import YOLO
import base64
import numpy as np
import cv2 as cv

app = Flask('YOLO Server')
modelo = YOLO('yolov8n.pt')

@app.route('/', methods=['GET', 'POST'])
def YOLOInference():
    try:
        data = request.json
        string64 = data['img']
        buffer = base64.b64decode(string64)
        imgArray = np.frombuffer(buffer, np.int8)
        img = cv.imdecode(imgArray, cv.IMREAD_UNCHANGED)
        result = modelo(img)
        return {'data': list(result[0].boxes.cls)}, 200
    except KeyError as ke:
        return {'error': f'Field {ke} missing'}, 400
    except Exception as e:
        return {'error': f'Exception {e}'}, 400
    
app.run(host='0.0.0.0')

