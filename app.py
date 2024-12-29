from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from utils.image_processing import process_image_patches, apply_segmentation, count_trees

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PATCH_FOLDER'] = 'static/patches/'
app.config['SEGMENTED_FOLDER'] = 'static/segmented/'
app.config['COUNTED_FOLDER'] = 'static/counted/'

# Route to display the UI
@app.route('/')
def index():
    return render_template('index.html')

# Route for image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and file.filename.endswith('.tif'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('process_patches', filename=file.filename))
    return 'Only .tiff files are allowed', 400

# Route to process patches and display result
@app.route('/process_patches/<filename>', methods=['GET', 'POST'])
def process_patches(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    patch_paths = process_image_patches(image_path, app.config['PATCH_FOLDER'])

    return render_template('patches.html', patches=patch_paths, filename=filename)

# Route to segment patches and detect orchard boundaries
@app.route('/segment/<filename>', methods=['POST'])
def segment_image(filename):
    selected_patches = request.form.getlist('selected_patches')  # Get the selected patches from the form

    if not selected_patches:
        return "No patches selected for segmentation", 400

    segmented_folder = app.config['SEGMENTED_FOLDER']
    segmented_paths = apply_segmentation(selected_patches, segmented_folder)
    
    return render_template('segmented.html', segmented=segmented_paths)

# @app.route('/segment/<filename>', methods=['POST'])
# def segment_image(filename):
#     patch_folder = app.config['PATCH_FOLDER']
#     segmented_folder = app.config['SEGMENTED_FOLDER']
#     segmented_paths = apply_segmentation(patch_folder, segmented_folder)
#     return render_template('segmented.html', segmented=segmented_paths)

# Route to count trees in patches
@app.route('/count_trees', methods=['POST'])
def count_trees_in_patches():
    segmented_folder = app.config['SEGMENTED_FOLDER']
    counted_folder = app.config['COUNTED_FOLDER']
    tree_counts = count_trees(segmented_folder, counted_folder)
    return render_template('tree_counts.html', counts=tree_counts)

if __name__ == '__main__':
    app.run()
