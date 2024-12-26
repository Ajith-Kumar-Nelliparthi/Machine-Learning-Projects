# Stroke Prediction
This project aims to predict the likelihood of a stroke based on various health parameters using machine learning techniques. \
The dataset used for this project is the "healthcare-dataset-stroke-data.csv" which contains information about patients' demographics, health conditions, and lifestyle choices.

## Attribute Information
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient

## Table of Contents

1. [Project Description](#project-description)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Local Setup](#local-setuo)
6. [Docker Setup](#docker-setup)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)

## Project Description
The goal of this project is to build a machine learning model that can predict the probability of a stroke based on input features \
such as age, gender, hypertension, heart disease, average glucose level,BMI, and smoking status. \
The project includes data preprocessing, model training, evaluation, and deployment using Flask and Docker.

## Repository Structure
```
├── Dockerfile
├── healthcare-dataset-stroke-data.csv
├── Pipfile
├── Pipfile.lock
├── stroke_load.py
├── stroke_predict.py
├── stroke_prediction (1).ipynb
├── stroke_prediction.py
└── README.md
```

→ Dockerfile: Docker configuration file for containerizing the application.\
→ healthcare-dataset-stroke-data.csv: The dataset used for training and testing the model.\
→ Pipfile and Pipfile.lock: Dependency management files for the project.\
→ stroke_load.py: Flask application for serving the model predictions.\
→ stroke_predict.py: Script for making predictions using the Flask API.\
→ stroke_prediction (1).ipynb: Jupyter notebook containing the data analysis and model training process.\
→ stroke_prediction.py: Python script for data preprocessing, model training, and evaluation.

## Prerequisites
Python 3.8 or higher
pipenv for managing dependencies
Docker (optional, for containerization)

## Installation
1.Clone the repository:
```
git clone https://github.com/Ajith-Kumar-Nelliparthi/Machine-Learning-Projects.git
cd Machine-Learning-Projects/Stroke_Prediction
```
2.Install dependencies using pipenv:
```
pipenv install
```
## Local Setup
1.Activate the virtual environment:
```
pipenv shell
```
2.Run the Flask application:
```
python stroke_load.py
```
![Screenshot 2024-12-26 210815](https://github.com/user-attachments/assets/92e80f13-70b9-4bbf-bd16-2c7c0c19cf79)

3.Make predictions using the Flask API:
```
python stroke_predict.py
```
![Screenshot 2024-12-26 210751](https://github.com/user-attachments/assets/24e87870-3378-4676-a8c8-f964ab11cbc9)

## Docker Setup
1.Build the Docker image:
```
docker build -t stroke-prediction .
```
2.Run the Docker container:
```
docker run -d -p 5000:5000 stroke-prediction
```
![Screenshot 2024-12-26 214920](https://github.com/user-attachments/assets/146d1293-e564-473c-8d27-158a4d478461)

3.Make predictions using the Flask API:
```
python stroke_predict.py
```
![Screenshot 2024-12-26 210751](https://github.com/user-attachments/assets/265d57f7-90d3-4b93-9eb8-99aaadcb07d3)

## Usage
The Flask API provides an endpoint /predict for making predictions. Send a POST request with the patient's data in JSON format to get the stroke probability.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
















































