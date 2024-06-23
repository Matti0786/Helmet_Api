
# Importing Libraries
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
model = YOLO('Models/model.pt')
import io
import base64
import numpy as np
import json
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        data = request.get_json()
        image_data = data['image']
        image_data = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_data))
        img_array = np.array(img)

        results = model(img_array)

        boxes = results[0].boxes
        names = results[0].names

        detections = []
        without_helmet_detected = False

        for box in boxes.data:
            x1, y1, x2, y2, confidence, class_index = box
            class_name = names[int(class_index)]
            detection = {
                'class': class_name,
            }
            detections.append(detection)

            # Check if any detection is 'Without Helmet'
            if class_name == 'Without Helmet':
                without_helmet_detected = True

        results = {
                'without_helmet_detected':without_helmet_detected
        }


        return results, 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    # app.run(host='0.0.0.0', port=5000, debug=True)
