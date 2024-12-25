# Medical Cost Prediction

This project aims to predict medical costs based on various factors such as age, sex, BMI, children, smoker, and region. Accurate prediction of medical costs can help in better planning and management of healthcare resources.
## Content
Columns\
1.age: age of primary beneficiary\
2.sex: insurance contractor gender, female, male\
3.bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,\
objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
4.children: Number of children covered by health insurance / Number of dependents\
5.smoker: Smoking\
6.region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.\
7.charges: Individual medical costs billed by health insurance

![image](https://github.com/user-attachments/assets/5f83da9c-81ff-4e83-b074-70773c5b45e7)


## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Data Description](#data-description)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Local Setup](#local-setup)
6. [Docker Setup](#docker-setup)
7. [Suggestions for Improvement](#suggestions-for-improvement)

## Repository Structure

```
Medical_Cost_Prediction/
├── data/
│   ├── expenses.csv                       # Dataset
├── notebooks/
│   ├── medical_cost_analysis.ipynb        # Exploratory Data Analysis notebook ,Model training and evaluation notebook
├── src/
│   ├── medical_cost_analysis.py           # Preprocessing scripts
│   ├── medical_cost_load.py               # Model training script
│   ├── medical_cost_predict.py            # Prediction script
├── Dockerfile                             # Docker configuration
├── requirements.txt                       # Python dependencies
├── README.md                              # Project README file
```

## Prerequisites
Ensure you have the following installed:
```
Python 3.8 or later
pip (Python package manager)
Docker (optional, for containerized setup)
```
## Installation
1.Clone the repository:
```
git clone https://github.com/Ajith-Kumar-Nelliparthi/Machine-Learning-Projects.git
cd Machine-Learning-Projects/Medical_Cost_Prediction
```
2.Install dependencies:
```
pip install -r requirements.txt
```
## Local Setup
1.Run the preprocessing script:
```
python src/medical_cost_analysis.py
```
2.Train the model:
```
python src/medical_cost_load.py
```
3.Make predictions:
```
python src/medical_cost_predict.py
```
## Docker Setup
1.Build the Docker image:
```
docker build -t medical-cost-prediction .
```
2.Run the Docker container:
```
docker run -it --rm -p 5000:5000 medical-cost-prediction
```
3.Access the application (if applicable) at http://localhost:5000.

## Suggestions for Improvement
Here are some suggestions to enhance the project:\
• Improve Documentation
Add more detailed descriptions of the features.\
Include visuals for exploratory data analysis (EDA) in the README or notebooks.\
• Model Interpretability\
Integrate SHAP or LIME for feature importance and explainability.\
• Scalability\
Deploy the model using Flask/FastAPI and serve it via a REST API.\
Provide a live demo or link to a hosted web app.\
• Testing\
Add unit tests for the preprocessing and training scripts.\
• Docker Enhancements\
Create a docker-compose.yml file to simplify multi-container setups.\
• Continuous Integration\
Set up CI/CD pipelines with GitHub Actions for automated testing and deployment.
