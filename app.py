import os
import uuid
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from src.inference import GearImagePredictor
from src.preprocessing import inspect_image_dataset

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize predictor
predictor = GearImagePredictor(model_path="models/gear_cnn_model.pt", model_name="mobilenet_v2")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    paths = {
        "undercut": os.path.abspath(os.path.join("dataset", "undercut")),
        "no_undercut": os.path.abspath(os.path.join("dataset", "no_undercut")),
        "model_file": os.path.abspath("models/gear_cnn_model.pt")
    }
    return render_template('index.html', paths=paths)

@app.route('/api/status', methods=['GET'])
def get_status():
    counts = inspect_image_dataset("dataset")
    is_trained = predictor.is_model_trained()
    return jsonify({
        "is_model_trained": is_trained,
        "model_name": predictor.model_name,
        "model_path": os.path.abspath(predictor.model_path),
        "dataset_counts": counts
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_gear():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty file uploaded."}), 400

    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file format. Only JPG and PNG are accepted."}), 400

    # Save uploaded file safely
    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"gear_{uuid.uuid4().hex[:8]}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    image_url = url_for('static', filename=f'uploads/{unique_filename}')

    # Run inference
    inference_result = predictor.predict(filepath)
    inference_result['image_url'] = image_url

    return jsonify(inference_result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f"Gear Tooth Undercutting Detector Web App Running!")
    print(f"URL: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False)
