# Gym Churn Probability
## Project Description
This project aims to predict the probability of gym members churning (i.e., not renewing their membership) using machine learning techniques. \
The dataset contains various features related to gym members' demographics and their gym usage patterns. The project involves data preprocessing, \
model training, evaluation, and deployment.

![image](https://github.com/user-attachments/assets/1039d529-1740-40b9-9d0b-86278f2cfd42)


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
Gym_Churn_Probability/
├── Dockerfile
├── gym_churn_prediction.ipynb
├── gym_churn_prediction.py
├── gym_churn_us.csv
├── gym_load.py
├── gym_predict.py
├── Pipfile
├── Pipfile.lock
├── requirements.txt
```

1) Dockerfile: Docker configuration file for containerizing the application.
2) gym_churn_us.csv: The dataset used for training and testing the model.
3) Pipfile and Pipfile.lock: Dependency management files for the project.
4) gym_load.py: Flask application for serving the model predictions.
5) gym_predict.py: Script for making predictions using the Flask API.
6) gym_churn_preiction.ipynb: Jupyter notebook containing the data analysis and model training process.
7) gym_churn_prediction.py: Python script for data preprocessing, model training, and evaluation.

## Prerequisites

- Python 3.13.0
- Pipenv
- Docker (optional, for Docker setup)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Ajith-Kumar-Nelliparthi/Machine-Learning-Projects.git
    cd Machine-Learning-Projects/Gym_Churn_Probability
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
    python gym_load.py
    ```
    ![Screenshot 2024-12-28 124509](https://github.com/user-attachments/assets/80807170-05a1-49ae-9f2a-56f20f1b5bc9)

3. In another terminal, run the prediction script:
    ```sh
    python gym_predict.py
    ```
   ![Screenshot 2024-12-28 125437](https://github.com/user-attachments/assets/b16f6a75-091a-4e9e-83d9-8fc67539d1eb)



## Docker Setup

1. Build the Docker image:
    ```sh
    docker build -t gym-churn_probability .
    ```
    

2. Run the Docker container:
    ```sh
    docker run -p 5000:5000 gym-churn-probability
    ```
    ![Screenshot 2024-12-27 170939](https://github.com/user-attachments/assets/f8cef12a-d64b-42b5-b2f2-84305e5e3327)



3. In another terminal, run the prediction script:
    ```sh
    python gym_predict.py
    ```
    ![Screenshot 2024-12-28 125437](https://github.com/user-attachments/assets/373f19ce-af20-43b7-8e1e-b4f6d2be40ba)

  

## Usage
After setting up the project, you can use the trained model to predict churn probabilities for new member data. \
Ensure that the input data is preprocessed similarly to the training data. You can create a script to load the model and pass new data for prediction.


## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License.
