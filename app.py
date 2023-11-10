from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    try:
        # Read the input image
        image = cv2.imread(image_path)

        # Perform image segmentation to get only the green area
        green_mask = get_green_mask(image)
        green_area = cv2.bitwise_and(image, image, mask=green_mask)

        # Estimate the LCC content based on color matching
        lcc_content = calculate_lcc_content(green_area)

        return lcc_content
    except Exception as e:
        return f"Error processing image: {str(e)}"

def get_green_mask(image):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for green color in HSV
    lower_green = np.array([20, 20, 20])  # Adjust these values based on your specific case
    upper_green = np.array([100, 255, 255]) # Adjust these values based on your specific case

    # Create a binary mask for the green area
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    return green_mask

def calculate_lcc_content(green_area):
    # Path to the folder containing the LCC score images
    scores_folder = '/home/ajit/Documents/LCC/colors'

    try:
        # Get a list of all files in the scores folder
        score_files = os.listdir(scores_folder)

        if not score_files:
            return "Error: No score images found in the folder."

        # Placeholder for the best match and its score
        best_match_score = None
        best_match_corr = 0

        for score_file in score_files:
            # Read the score image
            score_image = cv2.imread(os.path.join(scores_folder, score_file))

            # Resize the green_area to match the dimensions of the score_image
            height, width = score_image.shape[:2]
            green_area_resized = cv2.resize(green_area, (width, height))

            # Perform direct RGB matching
            diff = cv2.absdiff(green_area_resized, score_image)
            total_diff = np.sum(diff)

            # Update the best match if the current score is lower (less difference)
            if best_match_corr == 0 or total_diff < best_match_corr:
                best_match_corr = total_diff
                best_match_score = os.path.splitext(score_file)[0]  # Remove the file extension

        return best_match_score
    except Exception as e:
        return f"Error calculating LCC content: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', result="No file part")

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return render_template('index.html', result="No selected file")

        # If the file is allowed, save it and process the image
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the image and get the estimated LCC content
            lcc_content_result = process_image(file_path)

            # Display the result on the web page
            return render_template('index.html', result=f"Estimated LCC Score: {lcc_content_result}")

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
