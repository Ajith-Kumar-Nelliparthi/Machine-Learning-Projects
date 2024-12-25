### Water Potability

## Project Description
The Water Potability project aims to analyze water quality data to determine whether the water is safe for human consumption. The project utilizes machine learning techniques to classify water samples based on various chemical and physical properties. This analysis can help in identifying unsafe water sources and ensuring public health.

## Repository Structure
Water_Potability/
│
├── Water_potability_prediction_deployed/
│   ├── Dockerfile
│   ├── Pipfile
│   ├── Pipfile.lock
│   ├── water_potability_predict.py
│   ├── water_potability_load.py
│
├── requirements.txt
├── README.md

## Prerequisites
Python 3.13.0
pip
pipenv
Docker: If you plan to use Docker, ensure it is installed and running

## Local Setup
# 1.Clone the repository:
git clone https://github.com/Ajith-Kumar-Nelliparthi/Water_Potability.git
cd Water_Potability

# 2.Install dependencies:
pip install -r requirements.txt

# 3.Run the application:
python water_potability_predict.py

## Docker Setup
#Build the Docker image:
docker build -t water-potability .

# Run the Docker container:
docker run -p 5000:5000 water-potability

