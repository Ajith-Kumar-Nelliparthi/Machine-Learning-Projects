# Fetal Health Classification

Reduction of child mortality is reflected in several of the United Nations' Sustainable Development Goals and is a key indicator of human progress. The UN expects that by 2030, countries end preventable deaths of newborns and children under 5 years of age, with all countries aiming to reduce under‑5 mortality to at least as low as 25 per 1,000 live births.

Parallel to the notion of child mortality is maternal mortality, which accounts for 295,000 deaths during and following pregnancy and childbirth (as of 2017). The vast majority of these deaths (94%) occurred in low-resource settings, and most could have been prevented.

In light of what was mentioned above, Cardiotocograms (CTGs) are a simple and cost-accessible option to assess fetal health, allowing healthcare professionals to take action to prevent child and maternal mortality. The equipment itself works by sending ultrasound pulses and reading its response, thus shedding light on fetal heart rate (FHR), fetal movements, uterine contractions, and more.

## Data

This dataset contains 2126 records of features extracted from Cardiotocogram exams, which were then classified by three expert obstetricians into 3 classes:

- Normal
- Suspect
- Pathological

![dataset-card](https://github.com/user-attachments/assets/61b1ac32-e823-481c-b533-aaacbfa1b084)


## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Data Description](#data-description)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Local Setup](#local-setup)
6. [Docker Setup](#docker-setup)
7. [Suggestions for Improvement](#suggestions-for-improvement)

## Repository Structure

Fetal_Health_Classification/
├── data/
│   ├── fetal_health.csv       # Dataset
│   ├── processed_data.csv     # Preprocessed dataset (optional)
├── notebooks/
│   ├── EDA.ipynb              # Exploratory Data Analysis notebook
│   ├── Model_Training.ipynb   # Model training and evaluation notebook
├── src/
│   ├── preprocess.py          # Preprocessing scripts
│   ├── train.py               # Model training script
│   ├── predict.py             # Prediction script
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── README.md                  # Project README file
└── LICENSE                    # Project license




## Data Description

- `fetal_health.csv`: Contains the raw dataset with features extracted from Cardiotocogram exams.

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- pip (Python package manager)
- Docker (optional, for containerized setup)

## Installation

1. Clone the repository:
```sh
git clone https://github.com/Ajith-Kumar-Nelliparthi/Machine-Learning-Projects.git
cd Machine-Learning-Projects/Fetal_Health_Classification
```
2.Install dependencies:
```sh
pip install -r requirements.txt
```
## Local Setup
Run the preprocessing script:
```sh
python src/fetal_health.py
```
## Train the model:
```sh
python src/fetal_health_load.py
```
## Make predictions:
```sh
python src/fetal_health_predict.py
```
## Docker Setup
Build the Docker image:
```sh
docker build -t fetal-health-classification .
```
Run the Docker container:
```sh
docker run -it --rm -p 5000:5000 fetal-health-classification
```
![Screenshot 2024-12-24 210953](https://github.com/user-attachments/assets/83b43336-d269-46da-b81d-927df5c7c49d)

Access the application (if applicable) at http://localhost:5000.
## Suggestions for Improvement

Here are some suggestions to enhance the project:
[1.Improve Documentation]
Add more detailed descriptions of the features.
Include visuals for exploratory data analysis (EDA) in the README or notebooks.
[2.Model Interpretability]
Integrate SHAP or LIME for feature importance and explainability.
[3.Scalability]
Deploy the model using Flask/FastAPI and serve it via a REST API.
Provide a live demo or link to a hosted web app.
[4.Testing]
Add unit tests for the preprocessing and training scripts.
[5.Docker Enhancements]
Create a docker-compose.yml file to simplify multi-container setups.
[6.Continuous Integration]
Set up CI/CD pipelines with GitHub Actions for automated testing and deployment.



