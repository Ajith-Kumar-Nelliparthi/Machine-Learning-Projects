# Mushroom Classification Project

This project aims to classify mushrooms as either edible or poisonous based on various features. \
The dataset used for this project is the Mushroom Dataset from the UCI Machine Learning Repository.

![image](https://github.com/user-attachments/assets/1a67c1e2-f352-4e59-95c4-01b3900c7d31)


## Table of Contents

Project Description

1. [Project Description](#project-description)
1. [Repository Structure](#repository-structure)
2. [Prerequisites](#prerequisites)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Local Setup](#local-setup)
7. [Docker Setup](#docker-setup)
8. [Usage](#usage)
9. [Dataset](#dataset)
10. [Model](#Model)
11. [License](#License)
12. [Acknowledgements](#Acknowledgements)
13. [Contact](#Contact)


## Project Description

This project involves building a machine learning model to classify mushrooms as edible or poisonous \
based on their physical characteristics. The model is trained using the Mushroom Dataset from the UCI \
Machine Learning Repository. The project includes data preprocessing, model training, evaluation, and \
deployment using Flask and Docker.

## Repository Structure
```
Mushroom_Classification/
├── Dockerfile
├── mushroom_predict_notebook.ipynb
├── mushroom_predict_notebook.py
├── gym_churn_us.csv
├── mushroom_load.py
├── mushroom_predict.py
├── Pipfile
├── Pipfile.lock
├── requirements.txt
├── nn.py
```
## Prerequisites
1.Python 3.6 or higher
2.Pipenv
3.Docker (optional, for Docker setup)

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/Ajith-Kumar-Nelliparthi/Machine-Learning-Projects.git
    cd Machine-Learning-Projects/Mushroom_Classification
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
    python mushroom_load.py
    ```
    ![Screenshot 2025-01-10 114434](https://github.com/user-attachments/assets/7fe9f05b-8e6b-493a-ad10-f0ed25e57bca)


3. In another terminal, run the prediction script:
    ```sh
    python gym_predict.py
    ```
   ![Screenshot 2025-01-10 114425](https://github.com/user-attachments/assets/f3e85b93-0cd7-4f77-ab04-8496f23f7fab)



## Docker Setup

1. Build the Docker image:
    ```sh
    docker build -t mushroom_classification .
    ```
    

2. Run the Docker container:
    ```sh
    docker run -p 9999:9999 mushroom-classification
    ```
    ![Screenshot 2025-01-10 115846](https://github.com/user-attachments/assets/5ae712c1-f1b3-42cc-a6c9-a9378e1320c6)




3. In another terminal, run the prediction script:
    ```sh
    python mushroom_predict.py
    ```
    ![Screenshot 2025-01-10 115855](https://github.com/user-attachments/assets/09f5847e-edd9-45ba-8840-70f4c513ff6d)


## Usage
1. Training the Model
To train the model, you can use the Jupyter notebook mushroom_Predict_notebook.ipynb. This notebook includes \
data exploration, preprocessing, model training, and evaluation steps.

2. Running the Flask App
The Flask app serves the trained model and provides an endpoint for making predictions. To run the app, execute \
the following command:
```
  pipenv run python mushroom_load.py
```
3. The app will be available at http://localhost:9999/predict.

4. Making Predictions
You can use the mushroom_predict.py script to send a sample mushroom data to the Flask app and get a prediction. \
The script sends a POST request to the ```/predict``` endpoint with the mushroom features and prints the prediction result.

## Dataset
The dataset used in this project is the ```Mushroom Dataset``` from the UCI Machine Learning Repository.It contains 8124 \
instances of mushrooms, each described by 22 categorical features.

## Model
The model used in this project is a RandomForestClassifier from scikit-learn. The model is trained to classify mushrooms \
as either edible or poisonous based on their features.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements
1. The dataset is provided by the UCI Machine Learning Repository.
2. The project uses various Python libraries, including pandas, scikit-learn, Flask, and more.

## Contact
For any questions or inquiries, please contact ```Ajith Kumar Nelliparthi```. Feel free to customize this README file according to \
your project's specific details and requirements.























