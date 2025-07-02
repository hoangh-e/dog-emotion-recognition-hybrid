from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
import os
import json
import zipfile
import shutil
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid
from datetime import datetime
import tempfile

# Import prediction modules
from prediction_engine import PredictionEngine
from model_manager import ModelManager

app = Flask(__name__)
app.secret_key = 'dog_emotion_recognition_secret_key'

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
RESULTS_FOLDER = 'results'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
ALLOWED_MODEL_EXTENSIONS = {'pth', 'pt'}

# Create necessary directories
for folder in [UPLOAD_FOLDER, MODEL_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize managers
model_manager = ModelManager(MODEL_FOLDER)
prediction_engine = PredictionEngine()

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_model')
def add_model():
    return render_template('add_model.html')

@app.route('/predict_single')
def predict_single():
    models = model_manager.get_models()
    head_models = [m for m in models if m['type'] == 'Head-object-detection']
    classification_models = [m for m in models if m['type'] == 'Classification']
    return render_template('predict_single.html', 
                         head_models=head_models, 
                         classification_models=classification_models)

@app.route('/predict_batch')
def predict_batch():
    models = model_manager.get_models()
    head_models = [m for m in models if m['type'] == 'Head-object-detection']
    classification_models = [m for m in models if m['type'] == 'Classification']
    return render_template('predict_batch.html', 
                         head_models=head_models, 
                         classification_models=classification_models)

@app.route('/api/upload_model', methods=['POST'])
def upload_model():
    try:
        if 'model_file' not in request.files:
            return jsonify({'success': False, 'error': 'No model file selected'})
        
        file = request.files['model_file']
        model_name = request.form.get('model_name', '').strip()
        model_type = request.form.get('model_type', '')
        
        if not model_name:
            return jsonify({'success': False, 'error': 'Model name is required'})
        
        if not model_type:
            return jsonify({'success': False, 'error': 'Model type is required'})
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not allowed_file(file.filename, ALLOWED_MODEL_EXTENSIONS):
            return jsonify({'success': False, 'error': 'Invalid file type. Only .pth and .pt files allowed'})
        
        # Check if model name already exists
        if model_manager.model_exists(model_name):
            return jsonify({'success': False, 'error': 'Model name already exists'})
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(MODEL_FOLDER, filename)
        file.save(file_path)
        
        # Add model to manager
        model_manager.add_model(model_name, model_type, file_path, filename)
        
        return jsonify({'success': True, 'message': f'Model "{model_name}" uploaded successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict_single', methods=['POST'])
def api_predict_single():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file selected'})
        
        image_file = request.files['image']
        head_model_name = request.form.get('head_model')
        classification_model_name = request.form.get('classification_model')
        
        if not head_model_name or not classification_model_name:
            return jsonify({'success': False, 'error': 'Both head detection and classification models must be selected'})
        
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})
        
        if not allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'success': False, 'error': 'Invalid image type'})
        
        # Save uploaded image
        filename = secure_filename(image_file.filename)
        temp_image_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
        image_file.save(temp_image_path)
        
        # Get models
        head_model = model_manager.get_model(head_model_name)
        classification_model = model_manager.get_model(classification_model_name)
        
        if not head_model or not classification_model:
            return jsonify({'success': False, 'error': 'Selected models not found'})
        
        # Run prediction
        result = prediction_engine.predict_single(
            temp_image_path, 
            head_model, 
            classification_model
        )
        
        # Clean up temp file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        return jsonify({
            'success': True, 
            'result': result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict_batch', methods=['POST'])
def api_predict_batch():
    try:
        head_model_name = request.form.get('head_model')
        classification_model_name = request.form.get('classification_model')
        batch_type = request.form.get('batch_type')
        
        if not head_model_name or not classification_model_name:
            return jsonify({'success': False, 'error': 'Both models must be selected'})
        
        # Get models
        head_model = model_manager.get_model(head_model_name)
        classification_model = model_manager.get_model(classification_model_name)
        
        if not head_model or not classification_model:
            return jsonify({'success': False, 'error': 'Selected models not found'})
        
        image_paths = []
        temp_dirs = []
        
        if batch_type == 'folder':
            if 'folder_images' not in request.files:
                return jsonify({'success': False, 'error': 'No images selected'})
            
            files = request.files.getlist('folder_images')
            for file in files:
                if file.filename and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
                    filename = secure_filename(file.filename)
                    temp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
                    file.save(temp_path)
                    image_paths.append(temp_path)
                    temp_dirs.append(temp_path)
        
        elif batch_type == 'zip':
            if 'zip_file' not in request.files:
                return jsonify({'success': False, 'error': 'No zip file selected'})
            
            zip_file = request.files['zip_file']
            if zip_file.filename == '':
                return jsonify({'success': False, 'error': 'No zip file selected'})
            
            # Save and extract zip
            zip_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{secure_filename(zip_file.filename)}")
            zip_file.save(zip_path)
            
            extract_dir = os.path.join(UPLOAD_FOLDER, f"extracted_{uuid.uuid4()}")
            os.makedirs(extract_dir, exist_ok=True)
            temp_dirs.append(extract_dir)
            temp_dirs.append(zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find all images in extracted directory
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if allowed_file(file, ALLOWED_IMAGE_EXTENSIONS):
                        image_paths.append(os.path.join(root, file))
        
        elif batch_type == 'roboflow':
            roboflow_script = request.form.get('roboflow_script', '').strip()
            if not roboflow_script:
                return jsonify({'success': False, 'error': 'Roboflow script is required'})
            
            # Execute roboflow script and get images
            try:
                download_dir = os.path.join(UPLOAD_FOLDER, f"roboflow_{uuid.uuid4()}")
                os.makedirs(download_dir, exist_ok=True)
                temp_dirs.append(download_dir)
                
                # This would need to be implemented based on roboflow API
                # For now, return an error
                return jsonify({'success': False, 'error': 'Roboflow download not implemented yet'})
                
            except Exception as e:
                return jsonify({'success': False, 'error': f'Roboflow download failed: {str(e)}'})
        
        if not image_paths:
            return jsonify({'success': False, 'error': 'No valid images found'})
        
        # Run batch prediction
        results = prediction_engine.predict_batch(
            image_paths, 
            head_model, 
            classification_model
        )
        
        # Generate result file
        result_id = str(uuid.uuid4())
        result_file = os.path.join(RESULTS_FOLDER, f"batch_results_{result_id}.json")
        
        with open(result_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'head_model': head_model_name,
                'classification_model': classification_model_name,
                'total_images': len(image_paths),
                'results': results
            }, f, indent=2)
        
        # Clean up temp files
        for temp_path in temp_dirs:
            try:
                if os.path.isfile(temp_path):
                    os.remove(temp_path)
                elif os.path.isdir(temp_path):
                    shutil.rmtree(temp_path)
            except:
                pass
        
        return jsonify({
            'success': True, 
            'result_id': result_id,
            'total_images': len(image_paths),
            'download_url': f'/download_results/{result_id}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_results/<result_id>')
def download_results(result_id):
    result_file = os.path.join(RESULTS_FOLDER, f"batch_results_{result_id}.json")
    if os.path.exists(result_file):
        return send_file(result_file, as_attachment=True)
    else:
        return "Results not found", 404

@app.route('/api/models')
def api_models():
    return jsonify(model_manager.get_models())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 