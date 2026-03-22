from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
import io
import os
from PIL import Image as PILImage

app = Flask(__name__)
CORS(app)

print("Loading model...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'weed_classifier_resnet50.h5')
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ['broadleaf', 'grass', 'soil', 'soybean']
THRESHOLD = 0.70
print("Model loaded successfully!")

def preprocess_image(file_bytes):
    pil_img = PILImage.open(io.BytesIO(file_bytes)).convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(pil_img) / 255.0, axis=0).astype(np.float32)
    return pil_img, img_array

def pil_to_base64(pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def generate_gradcam(img_array, pil_img):
    last_conv_layer = 'conv5_block3_out'
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, list):
            predictions = predictions[0]
        preds_array = predictions[0].numpy()
        pred_class_idx = int(np.argmax(preds_array))
        loss = predictions[0][pred_class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_out = conv_outputs[0].numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_out[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(conv_out, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)
    return img_cv, heatmap_colored, overlay

def img_to_base64(img_cv):
    _, buffer = cv2.imencode('.jpg', img_cv)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'model': 'weed_classifier_resnet50',
        'classes': CLASS_NAMES,
        'threshold': THRESHOLD
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    try:
        file_bytes = file.read()

        # Preview — works for TIF JPG PNG all formats
        preview_pil = PILImage.open(io.BytesIO(file_bytes)).convert('RGB')
        preview_b64 = pil_to_base64(preview_pil)

        # Preprocess for model
        pil_img, img_array = preprocess_image(file_bytes)

        # Predict
        raw_preds = model.predict(img_array)
        if isinstance(raw_preds, list):
            raw_preds = raw_preds[0]
        preds_array = np.array(raw_preds).flatten()

        pred_idx = int(np.argmax(preds_array))
        confidence = float(preds_array[pred_idx]) * 100
        max_conf_ratio = float(preds_array[pred_idx])

        if max_conf_ratio < THRESHOLD:
            predicted_class = "Uncertain"
            is_uncertain = True
        else:
            predicted_class = CLASS_NAMES[pred_idx]
            is_uncertain = False

        # Grad-CAM
        orig_cv, heatmap_cv, overlay_cv = generate_gradcam(img_array, pil_img)

        # All predictions
        all_preds = [
            {'class': CLASS_NAMES[i], 'confidence': round(float(preds_array[i]) * 100, 1)}
            for i in range(len(CLASS_NAMES))
        ]

        recommendations = {
            'soybean': 'Healthy soybean crop detected. Continue regular monitoring and maintain optimal irrigation.',
            'broadleaf': 'Broadleaf weed detected. Apply selective herbicide. Remove manually if infestation is low.',
            'grass': 'Grass weed detected. Use grass-selective herbicide. Monitor spread to neighboring areas.',
            'soil': 'Bare soil detected. Consider cover cropping to prevent erosion and nutrient loss.',
            'Uncertain': 'Model is not confident. Please upload a clearer field image.'
        }

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': round(confidence, 1),
            'is_uncertain': is_uncertain,
            'all_predictions': all_preds,
            'preview': preview_b64,
            'recommendation': recommendations.get(predicted_class, 'Consult an agricultural expert.'),
            'gradcam': {
                'original': img_to_base64(orig_cv),
                'heatmap': img_to_base64(heatmap_cv),
                'overlay': img_to_base64(overlay_cv)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')