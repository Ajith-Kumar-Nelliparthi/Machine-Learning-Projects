# Project Description

Reduction of child mortality is reflected in several of the United Nations' Sustainable Development Goals and is a key indicator of human progress.
The UN expects that by 2030, countries end preventable deaths of newborns and children under 5 years of age, with all countries aiming to reduce under‑5 mortality to at least as low as 25 per 1,000 live births.

Parallel to notion of child mortality is of course maternal mortality, which accounts for 295 000 deaths during and following pregnancy and childbirth (as of 2017). The vast majority of these deaths (94%) occurred in low-resource settings, and most could have been prevented.

In light of what was mentioned above, Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality. The equipment itself works by sending ultrasound pulses and reading its response, thus shedding light on fetal heart rate (FHR), fetal movements, uterine contractions and more.

## Data
This dataset contains 2126 records of features extracted from Cardiotocogram exams, which were then classified by three expert obstetritians into 3 classes:

-- Normal
-- Suspect
-- Pathological

(https://storage.googleapis.com/kaggle-datasets-images/916586/1553068/ddd9373754b16217a54a513f0d94628a/dataset-card.png?t=2020-10-12-00-50-47)



# Table of Contents

1. Repository Structure
2. Data Description
3. Prerequisites
4. Installation
5. Local Setup
6. Docker Setup
7. Suggestions for Improvement

# Repository Structure

Fetal_Health_Classification/
├── data/
│   ├── fetal_health.csv       # Dataset
├── notebooks/
│   ├── fetal_health_prediction.ipynb         # Exploratory Data Analysis notebook
│   ├── fetal_health.py                       # Model training and evaluation notebook
├── src/
│   ├── fetal_health_load.py               # Model training script
│   ├── fetal_health_predict.py            # Prediction script
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── README.md                  # Project README file
└── LICENSE                    # Project license

# Target:
fetal_health - Categorical label indicating the health status.

# Prerequisites
Ensure you have the following installed:
Python 3.8 or later
pip (Python package manager)
Docker (optional, for containerized setup)

# Installation
## 1.Clone the repository:
git clone https://github.com/Ajith-Kumar-Nelliparthi/Machine-Learning-Projects.git
cd Machine-Learning-Projects/Fetal_Health_Classification

## 2.Install dependencies:
pip install -r requirements.txt

# Local Setup
## Run the preprocessing script:
python src/fetal_health.py
## Train the model:
python src/fetal_health_load.py
## Make predictions:
python src/fetal_healthpredict.py

# Docker Setup
## 1.Build the Docker image:
docker build -t fetal-health-classification .

## Run the Docker container:
docker run -it --rm -p 5000:5000 fetal-health-classification
## Access the application (if applicable) at http://localhost:5000.

# Suggestions for Improvement
Here are some suggestions to enhance the project:

## Improve Documentation:
Add more detailed descriptions of the features.
Include visuals for exploratory data analysis (EDA) in the README or notebooks.
## Model Interpretability:
Integrate SHAP or LIME for feature importance and explainability.
## Scalability:
Deploy the model using Flask/FastAPI and serve it via a REST API.
Provide a live demo or link to a hosted web app.
## Testing:
Add unit tests for the preprocessing and training scripts.
## Docker Enhancements:
Create a docker-compose.yml file to simplify multi-container setups.
## Continuous Integration:
Set up CI/CD pipelines with GitHub Actions for automated testing and deployment.


# Project Description

Reduction of child mortality is reflected in several of the United Nations' Sustainable Development Goals and is a key indicator of human progress.
The UN expects that by 2030, countries end preventable deaths of newborns and children under 5 years of age, with all countries aiming to reduce under‑5 mortality to at least as low as 25 per 1,000 live births.

Parallel to notion of child mortality is of course maternal mortality, which accounts for 295 000 deaths during and following pregnancy and childbirth (as of 2017). The vast majority of these deaths (94%) occurred in low-resource settings, and most could have been prevented.

In light of what was mentioned above, Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality. The equipment itself works by sending ultrasound pulses and reading its response, thus shedding light on fetal heart rate (FHR), fetal movements, uterine contractions and more.

## Data
This dataset contains 2126 records of features extracted from Cardiotocogram exams, which were then classified by three expert obstetritians into 3 classes:

-- Normal
-- Suspect
-- Pathological

(https://storage.googleapis.com/kaggle-datasets-images/916586/1553068/ddd9373754b16217a54a513f0d94628a/dataset-card.png?t=2020-10-12-00-50-47)



# Table of Contents

1. Repository Structure
2. Data Description
3. Prerequisites
4. Installation
5. Local Setup
6. Docker Setup
7. Suggestions for Improvement

# Repository Structure

Fetal_Health_Classification/
├── data/
│   ├── fetal_health.csv       # Dataset
├── notebooks/
│   ├── fetal_health_prediction.ipynb         # Exploratory Data Analysis notebook
│   ├── fetal_health.py                       # Model training and evaluation notebook
├── src/
│   ├── fetal_health_load.py               # Model training script
│   ├── fetal_health_predict.py            # Prediction script
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── README.md                  # Project README file
└── LICENSE                    # Project license

# Target:
fetal_health - Categorical label indicating the health status.

# Prerequisites
Ensure you have the following installed:
Python 3.8 or later
pip (Python package manager)
Docker (optional, for containerized setup)

# Installation
## 1.Clone the repository:
git clone https://github.com/Ajith-Kumar-Nelliparthi/Machine-Learning-Projects.git
cd Machine-Learning-Projects/Fetal_Health_Classification

## 2.Install dependencies:
pip install -r requirements.txt

# Local Setup
## Run the preprocessing script:
python src/fetal_health.py
## Train the model:
python src/fetal_health_load.py
## Make predictions:
python src/fetal_healthpredict.py

# Docker Setup
## 1.Build the Docker image:
docker build -t fetal-health-classification .

## Run the Docker container:
docker run -it --rm -p 5000:5000 fetal-health-classification
## Access the application (if applicable) at http://localhost:5000.

# Suggestions for Improvement
Here are some suggestions to enhance the project:

## Improve Documentation:
Add more detailed descriptions of the features.
Include visuals for exploratory data analysis (EDA) in the README or notebooks.
## Model Interpretability:
Integrate SHAP or LIME for feature importance and explainability.
## Scalability:
Deploy the model using Flask/FastAPI and serve it via a REST API.
Provide a live demo or link to a hosted web app.
## Testing:
Add unit tests for the preprocessing and training scripts.
## Docker Enhancements:
Create a docker-compose.yml file to simplify multi-container setups.
## Continuous Integration:
Set up CI/CD pipelines with GitHub Actions for automated testing and deployment.


