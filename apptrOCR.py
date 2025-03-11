from flask import Flask, render_template, request, flash, redirect, url_for, send_file
import os
import cv2
import numpy as np
from inferenceTrOCR import predict_text  # TrOCR Model Inference Function
from PIL import Image

# Flask app configuration
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

UPLOAD_FOLDER = 'uploads'
LINE_FOLDER = os.path.join(UPLOAD_FOLDER, 'lines')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)   
os.makedirs(LINE_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image: Convert to grayscale and apply thresholding
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary, image

# Segment lines from the uploaded document
def segment_lines(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilation to connect words into lines
    kernel = np.ones((3, 100), np.uint8)  # Adjust kernel size based on text densityy
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours of lines
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])  # Sort lines by Y-axis

    
    extracted_text = []
    

    for i, line in enumerate(sorted_lines):
        x, y, w, h = cv2.boundingRect(line)
        line_roi = image[y:y+h, x:x+w]  # Crop the detected line

        line_pil = Image.fromarray(cv2.cvtColor(line_roi, cv2.COLOR_BGR2RGB))

        # Pass segmented line to TrOCR model
        predicted_text, confidence = predict_text(line_pil)  # Pass directly instead of saving

        extracted_text.append(f"{predicted_text} (Confidence: {round(confidence, 2)}%)")

    return extracted_text

#@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('home'))
    return render_template('login.html')

#@app.route('/home')
@app.route('/')
def home():
    return render_template('base.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Process and segment the imageeee
            extracted_text = segment_lines(file_path)
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('home'))
        
        clean_text = "\n".join([line.split(" (Confidence:")[0] for line in extracted_text])
        return render_template('onnxPrediction.html',image_filename=filename, extracted_text=extracted_text, clean_text=clean_text)

    flash('Allowed file types: png, jpg, jpeg')
    return redirect(url_for('home'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/download-text')
def download_text():
    return send_file(os.path.join(LINE_FOLDER, "extracted_text.txt"), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
