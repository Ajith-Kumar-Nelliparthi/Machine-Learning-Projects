# 📌 Project Overview

Lung cancer is one of the most common and deadliest types of cancer. Early detection through medical imaging can significantly improve survival rates. This project aims to build a machine learning model to classify lung cancer based on imaging data, helping in early diagnosis and treatment planning.


https://github.com/user-attachments/assets/15d1370a-ebc9-4a7b-b8a1-67286b6dee71



# About Dataset
This [dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) contains 25,000 histopathological images with 5 classes. All images are 768 x 768 pixels in size and are in jpeg file format.
The images were generated from an original sample of HIPAA compliant and validated sources, consisting of 750 total images of lung tissue (250 benign lung tissue, 250 lung adenocarcinomas, and 250 lung squamous cell carcinomas) and 500 total images of colon tissue (250 benign colon tissue and 250 colon adenocarcinomas) and augmented to 25,000 using the Augmentor package.
There are five classes in the dataset, each with 5,000 images, being:

Lung benign tissue\
Lung adenocarcinoma\
Lung squamous cell carcinoma\
Colon adenocarcinoma\
Colon benign tissue

## Project Structure
```
Lungt-Cancer-Classification/
├──app.py
├──Dockerfile
├──lung_cancer_classification.ipynb
├──Pipfile
├──Pipfile.lock
├──predict.py
├──requirements.txt
├──static/
│   └──   style.css
├──templates/
│   └──    index.html
```
## Features
1. Data Preprocessing: Clean and prepare the dataset for analysis.
2. Model Training: Implement various keras pre-trained models to train the model.
3. Evaluation: Assess the model's performance using metrics such as accuracy, precision, and recall.
4. Visualization: Provide visual insights into the data and model performance.

## Setup and Installation
## Prerequisites
Python 3.11\
Docker (optional, for containerized deployment)
## Installation
1. Clone the repository:
```
  git clone https://github.com/Ajith-Kumar-Nelliparthi/Machine-Learning-Projects.git
  cd Lung-Cancer-Classification

```
2. Install dependencies using Pipenv:
```
  pipenv install
```
3. Activate the virtual environment:
```
  pipenv shell
```
4. Install additional dependencies:
```
  pip install -r requirements.txt
```
## Running the Application
##Using Flask
1. Run the Flask application:
```
python app.py
```
![Screenshot 2025-02-17 211739](https://github.com/user-attachments/assets/1f65b82b-538e-4625-8462-9df77bc57e1f)
2. Open your web browser and navigate to ```http://localhost:8000```.

![Screenshot 2025-02-17 214837](https://github.com/user-attachments/assets/1419f92c-8f4a-4c6d-8fc0-5d0ef140d113)

## Using Docker
1. Build the Docker image:
```
docker build -t lung-cancer-detection .
```
3. Run the Docker container:
```
docker run -p 8000:8000 lung-cancer-detection
```
![Screenshot 2025-02-18 082904](https://github.com/user-attachments/assets/0da61248-1117-4ca2-b49c-1cc09d97f3d8)


3.Open your web browser and navigate to ```http://localhost:8000```.
![Screenshot 2025-02-18 074636](https://github.com/user-attachments/assets/47187ae3-aa77-4c4f-92a1-8f0e08fe5940)


## Usage
1. Open the web application in your browser.
2. Upload a lung image using the file input.
3. Click the "Predict" button to get the prediction.
4. The result will display the predicted class and the corresponding class name.
## Model Training
The model is trained using the dataset located in the lung_colon_image_set directory. The training process is documented in the Jupyter notebooks:
1. lung_cancer_classification.ipynb

## File Descriptions
• app.py: The main Flask application file.\
• Dockerfile: Docker configuration file for containerizing the application.\
• lung_cancer_classifier_model.h5: Pre-trained model file.\
• lung_cancer_classification.ipynb: Jupyter notebook for model training and evaluation.\
• predict.py: Script for making predictions using the trained model.\
• requirements.txt: List of Python dependencies.\
• style.css: CSS file for styling the web interface.\
• index.html: HTML template for the web interface.

##Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.


## Acknowledgements

1. TensorFlow for providing the machine learning framework.
2. Flask for the web framework.
3. The dataset used for training the model.

## 📧 Contact

For any queries, reach out to:

❗ GitHub: [Ajith-Kumar-Nelliparthi](https://github.com/Ajith-Kumar-Nelliparthi)\
❗ Linkedin : [Ajith Kumar Nelliparthi](https://www.linkedin.com/in/nelliparthi-ajith-233803262)\
❗ Twitter : [Ajith Kumar Nelliparthi](https://x.com/Ajith532542840)

° Feel free to customize this README file according to your project's specific details and requirements.

















