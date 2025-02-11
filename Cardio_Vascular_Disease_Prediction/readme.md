# Description

This heart disease dataset is acquired from one of the multispecialty hospitals in India. Over 14 common features make \
it one of the heart disease datasets available so far for research purposes. This dataset consists of 1000 subjects with \
12 features. This dataset will be useful for building early-stage heart disease detection as well as for generating predictive machine-learning models.

![image](https://github.com/user-attachments/assets/c7ce8aae-d8df-43d1-80f1-c2172bd96a98)


The task is to predict the presence or absence of cardiovascular disease (CVD) using the patient examination results.

## Data description

There are 3 types of input features:
1. Objective: factual information;
2. Examination: results of medical examination;
3. Subjective: information given by the patient.

## Features:

• Age | Objective Feature | age | int (days)\
• Height | Objective Feature | height | int (cm) |\
• Weight | Objective Feature | weight | float (kg) |\
• Gender | Objective Feature | gender | categorical code |\
• Systolic blood pressure | Examination Feature | ap_hi | int |\
• Diastolic blood pressure | Examination Feature | ap_lo | int |\
• Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |\
• Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |\
• Smoking | Subjective Feature | smoke | binary |\
• Alcohol intake | Subjective Feature | alco | binary |\
• Physical activity | Subjective Feature | active | binary |\
• Presence or absence of cardiovascular disease | Target Variable | cardio | binary |

## Project Description

This project aims to predict the probability of cardiovascular disease in patients using machine learning models. \
The project utilizes various machine learning algorithms to train models on a dataset of patient health records \
and provides a REST API for making predictions.

## Table of Contents

- [Project Description](#project-description)
- [Table of Contents](#table-of-contents)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Local Setup](#local-setup)
- [Docker Setup](#docker-setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Repository Structure
```
Cardio_Vascular_Disease_Prediction/ 
├── pycache/ 
├── Dataset
| ├── cardio_train.csv
├── Deplotment
| ├── cardio_load.py 
| ├── cardio_predict.py 
| ├── cardio_disease_prediction.py 
| ├── Cardio_disease_prediction.ipynb 
| ├── cardio_xgb.pkl 
| ├── Dockerfile 
| ├── Pipfile 
| ├── Pipfile.lock
├──neural_nn.ipynb
├── neural.py
├── requirements.txt
└── README.md
```
1) Dockerfile: Docker configuration file for containerizing the application.
2) cardio_train.csv: The dataset used for training and testing the model.
3) Pipfile and Pipfile.lock: Dependency management files for the project.
4) cardio_load.py: Flask application for serving the model predictions.
5) cardio_predict.py: Script for making predictions using the Flask API.
6) cardio_disease_preiction.ipynb: Jupyter notebook containing the data analysis and model training process.
7) cardio_disease_prediction.py: Python script for data preprocessing, model training, and evaluation.


## Prerequisites

- Python 3.13.0
- Pipenv
- Docker (optional, for Docker setup)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Ajith-Kumar-Nelliparthi/Machine-Learning-Projects.git
    cd Machine-Learning-Projects/Cardio_Vascular_Disease_Prediction
    ```

2. Install dependencies using Pipenv:
    ```sh
    pip install pipenv
    pipenv install
    ```

## Local Setup

1. Activate the virtual environment:
    ```sh
    pipenv shell
    ```

2. Run the Flask app:
    ```sh
    python cardio_load.py
    ```
    ![Screenshot 2024-12-27 165750](https://github.com/user-attachments/assets/e8709faf-45fa-48f8-86fd-7c493fc4ffda)


3. In another terminal, run the prediction script:
    ```sh
    python cardio_predict.py
    ```
    ![Screenshot 2024-12-27 165800](https://github.com/user-attachments/assets/9547cdca-2cd5-42be-9299-805e8b7e83de)


## Docker Setup

1. Build the Docker image:
    ```sh
    docker build -t cardio-disease-prediction .
    ```

2. Run the Docker container:
    ```sh
    docker run -p 9999:9999 cardio-disease-prediction
    ```
    ![Screenshot 2024-12-27 170939](https://github.com/user-attachments/assets/1ce8afcc-2af6-4ffc-8549-ce1a0150bf07)


3. In another terminal, run the prediction script:
    ```sh
    python cardio_predict.py
    ```
    ![Screenshot 2024-12-27 165800](https://github.com/user-attachments/assets/2e640320-496e-44b9-9558-b3818100be3f)


## Usage

- The Flask app provides a REST API endpoint at `/predict` for making predictions.
- Send a POST request with patient data in JSON format to get the probability of cardiovascular disease.

Example:
```json
{
  "id": 989,
  "age": 68,
  "gender": 1,
  "height": 155,
  "weight": 69.0,
  "ap_hi": 130,
  "ap_lo": 80,
  "cholesterol": 2,
  "gluc": 1,
  "smoke": 1,
  "alco": 1,
  "active": 1
}
```
## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License.


