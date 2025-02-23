from flask import Flask, render_template, request, flash, redirect, url_for, send_file, jsonify, send_from_directory
import os
import numpy as np
import cv2
import onnxruntime as ort
from io import BytesIO
from PIL import Image

# Import ONNX inference function
from inferenceOnnx import predict_text

app = Flask(__name__)

# Secret key for flash messages
app.secret_key = 'your-secret-key-here'

# Configure upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Run ONNX model inference
        try:
            predicted_text, confidence = predict_text(file_path)
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('home'))
        
        #return jsonify({
        #    'filename': filename,
        #    'predicted_text': predicted_text,
        #    'confidence': confidence
        #})
        return render_template('onnxPrediction.html', 
                               image_filename=filename, 
                               predicted_text=predicted_text, 
                               confidence_score=round(confidence, 2))
    
    flash('Allowed file types are: png, jpg, jpeg')
    return redirect(url_for('home'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
