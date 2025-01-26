# Sunglass Rack Vacancy Detection System

## Overview
The Sunglass Rack Vacancy Detection System is an automated solution to detect and analyze rack vacancy for sunglass displays. The system leverages a custom-trained YOLOv5 model and image processing techniques to identify and count sunglass racks, as well as to measure the vacancy percentage of each rack. This tool can process bulk images, generate detailed rack vacancy reports, and can be utilized for monitoring sunglass rack inventory in retail environments.

The model was trained using labeled data from Roboflow and is capable of detecting racks in different lighting conditions, handling varying image resolutions, and producing high accuracy in terms of detected rack vacancy.

## Key Features
- **Automated Rack Detection**: Using a YOLOv5 model to detect sunglass racks in images.
- **Vacancy Analysis**: Percentage of vacant space in the racks is calculated based on white pixel detection.
- **Bulk Image Processing**: Multiple images can be processed in batch, generating comprehensive reports for each batch.
- **Customizable Output**: The generated report includes detailed rack vacancy information, which can be exported to CSV format.
- **Flexible Deployment**: The solution is built using Flask and can be deployed easily as a web service for user interaction.

## Installation

### Prerequisites
Before you run the system, make sure you have the following installed:

1. **Python 3.7+**: The system is built with Python 3.7 and above.
2. **Required Libraries**: Install the required Python packages using the following command:
    ```bash
    pip install -r requirements.txt
    ```

3. **Torch and YOLOv5**: The project uses YOLOv5 for object detection. It is compatible with both CPU and GPU usage.
4. **Roboflow API**: The system utilizes the Roboflow platform for labeling images.

## How It Works

### Image Upload and Processing
1. **Upload Images**: Images are uploaded through the Flask web interface.
2. **Image Preprocessing**: The uploaded image is preprocessed and passed through the YOLOv5 model for rack detection.
3. **Vacancy Calculation**: The system calculates the percentage of vacant space in each detected rack by analyzing the white pixels.
4. **Report Generation**: A CSV report is generated with detailed statistics of rack vacancy, including the number of racks detected and the vacancy percentage.

### Image Processing and Vacancy Detection

1. **YOLOv5 Detection**: The YOLOv5 model is trained to detect sunglass racks within an image. Each detected rack's bounding box coordinates are identified.
   
2. **Vacancy Calculation**: Once racks are detected, vacancy is calculated by counting the white pixels in each rack's cropped image. The system uses the following techniques:
   - **Thresholding**: Binarizes the image to isolate white pixels (vacant space).
   - **Pixel Count**: Counts the white pixels in each detected area to calculate the vacancy percentage.
   
3. **Result Compilation**: The system compiles the results into a CSV file, which includes:
   - S.No.
   - Date of Processing
   - Filename
   - Rack Vacancy Percentage
   - Number of Racks Detected

## Example of Generated Report

The generated report will have the following columns:

| S.No. | Date       | Filename       | % of Rack Vacancy | No of Racks Detected |
|-------|------------|----------------|-------------------|----------------------|
| 1     | 2025-01-26 | image1.jpg     | 45.6              | 4                    |
| 2     | 2025-01-26 | image2.jpg     | 12.3              | 5                    |

## Model Training and Labeling

The YOLOv5 model was trained on custom-labeled data, which was labeled using **Roboflow**. Roboflow allows easy annotation of images for object detection tasks, and its API is integrated into the project to fetch labeled datasets for training.

### YOLOv5 Model
- The model is trained to detect sunglass racks in various display conditions.
- The custom-trained model (`best_new_model.pt`) is used in this project to predict the location of racks in the uploaded images.

### Roboflow Integration
1. **Labeling**: Images are manually labeled in Roboflow to mark sunglass racks.
2. **Model Training**: The labeled data is used to train a YOLOv5 model on Roboflow’s platform.
3. **Model Export**: Once the model is trained, it can be exported in `.pt` format (PyTorch model).

## Deployment and Usage
### Flask Web Interface
The Flask app provides a simple web interface where users can:
- Upload images for processing.
- View processed images with highlighted detected racks.
- Download the generated report in CSV format.

1. **Upload Images**: Users upload images directly on the web interface.
2. **Process Images**: Upon clicking the "Process" button, the images are passed to the backend, where the processing and vacancy detection takes place.
3. **View Results**: The results are displayed, showing the processed image along with the rack vacancy percentage.

## Future Enhancements
- **Real-Time Processing**: Enhance the system to support live camera feed processing.
- **User Authentication**: Add user login for better security and report tracking.
