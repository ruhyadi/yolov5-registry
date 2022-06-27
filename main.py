import json
import base64
from PIL import Image
import io

from model_handler import load_model

def init_context(context):
    context.logger.info("Init context...  0%")

    # Read the DL model
    # TODO: load your model with model_hander utils
    model = load_model(model_path='weights/yolov5s.pt')
    context.user_data.model = model

    context.logger.info("Init context...100%")

def handler(context, event):
    # TODO: change YOUR_MODEL
    context.logger.info("Run YOUR_MODEL")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    # TODO: change threshold value, default: 0.5
    threshold = float(data.get("threshold", 0.5))
    context.user_data.model.conf = threshold
    image = Image.open(buf)
    
    # TODO: change inference mechanism
    results = context.user_data.model(image)

    encoded_results = []
    for result in results:
        encoded_results.append({
            'confidence': result['confidence'],
            'label': result['name'],
            'points': [
                result['xmin'],
                result['ymin'],
                result['xmax'],
                result['ymax']
            ],
            'type': 'rectangle'
        })

    return context.Response(body=json.dumps(encoded_results), headers={},
        content_type='application/json', status_code=200)
