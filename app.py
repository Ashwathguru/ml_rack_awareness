from flask import Flask, render_template, request, url_for
import os
import shutil
from image_processing import run  # Import your processing function
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/'
app.config['PROCESSED_IMAGE_FOLDER'] = 'runs/detect/'
app.config['RUN_IMAGE_FOLDER'] = 'static/runs/'
app.config['REPORT_FOLDER'] = 'run_reports/'

# Register the 'runs' folder as a static folder
app.static_folder = 'static'
app.add_url_rule('/runs/<path:filename>', endpoint='runs', view_func=app.send_static_file)

# Helper function to clean directories
def clean_directory(folder_path):
    folder_name = os.path.basename(folder_path)  # Get the folder name
    if os.path.exists(folder_path):
        if folder_name == 'exp':
            shutil.rmtree(folder_path)  # Delete the entire folder if it is named 'exp'
        else:
            for file_or_dir in os.listdir(folder_path):  # Delete files only for other folders
                file_or_dir_path = os.path.join(folder_path, file_or_dir)
                if os.path.isfile(file_or_dir_path):
                    os.remove(file_or_dir_path)
                elif os.path.isdir(file_or_dir_path):
                    shutil.rmtree(file_or_dir_path)



# Route for the upload page
@app.route('/')
def upload_page():
    # Clean up directories after rendering
    clean_directory(app.config['PROCESSED_IMAGE_FOLDER'])
    clean_directory(app.config['UPLOAD_FOLDER'])
    clean_directory(app.config['REPORT_FOLDER'])
    clean_directory(app.config['RUN_IMAGE_FOLDER'])
    return render_template('upload.html')

# Route for processing the image
@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No file uploaded", 400

    # Save uploaded image
    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Run the image processing function
    run(source=app.config['UPLOAD_FOLDER'], destination=app.config['REPORT_FOLDER'])

    report_path = app.config['REPORT_FOLDER']

    # Get results from the latest report
    report_file = sorted(os.listdir(report_path))[-1]
    report_df = pd.read_csv(os.path.join(report_path, report_file))
    last_entry = report_df.iloc[-1]
    racks_detected = last_entry['No of Racks Detected']
    rack_vacancy = last_entry['% of Rack Vacany']



    # Move all contents from 'runs/detect/exp' to 'static/runs'
    for item in os.listdir('runs/detect/exp'):
        shutil.move(os.path.join('runs/detect/exp', item), os.path.join('static/runs', item))

    # Render result page
    return render_template(
        'result.html',
        uploaded_image=file.filename,
        processed_image=file.filename,
        racks_detected=racks_detected,
        rack_vacancy=rack_vacancy,
    )

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
