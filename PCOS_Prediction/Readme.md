# Polycystic Ovary Syndrome (PCOS) Classification

This dataset simulates the prevalence and key risk factors of Polycystic Ovary Syndrome (PCOS) among women of reproductive age (15–49 years) across the world's top 75 most 
populous countries. The dataset consists of 120,000 records and 17 variables, reflecting real-world patterns and variability in PCOS diagnosis and related factors.
The dataset is used for exploratory data analysis (EDA) and to develop classification models for predicting the likelihood of undiagnosed PCOS based on various risk factors.

![image](https://github.com/user-attachments/assets/e9be017c-8de9-4f16-87f8-4f82f0f39e3c)

## Table of Contents
- [Project Description](#project-description)
- [Dataset Overview](#dataset-overview)
- [Table of Contents](#table-of-contents)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Local Setup](#local-setup)
- [Docker Setup](#docker-setup)

## Project Description
The PCOS Prediction project aims to provide a machine learning model that predicts the likelihood of Polycystic Ovary Syndrome (PCOS) based on various user inputs. 
The application is built using Flask for the backend and serves a web interface for user interaction.

## Dataset Overview
Describe the columns and their data types.

Example: The dataset includes the following columns:

1. Country: The country of origin (Categorical).
2. Age: Age of the individual (Numerical).
3. BMI: Body Mass Index category (Categorical: Overweight, Normal, Underweight).
4. Menstrual Regularity: Whether menstrual cycles are regular or irregular (Categorical).
5. Hirsutism: The severity of excess body hair growth (Categorical).
6. Acne Severity: Acne severity (Categorical).
7. Family History of PCOS: Whether the individual has a family history of PCOS (Categorical: Yes/No).
8. Insulin Resistance: Whether insulin resistance is present (Categorical: Yes/No).
9. Lifestyle Score: A score reflecting lifestyle habits (Numerical).
10. Stress Levels: Stress levels (Numerical).
11. Urban/Rural: Whether the individual lives in an urban or rural area (Categorical).
12. Socioeconomic Status: Socioeconomic status of the individual (Categorical).
13. Awareness of PCOS: Whether the individual is aware of PCOS (Categorical: Yes/No).
14. Fertility Concerns: Whether the individual has fertility concerns (Categorical: Yes/No).
15. Undiagnosed PCOS Likelihood: The likelihood of having undiagnosed PCOS (Numerical).
16. Ethnicity: Ethnic background (Categorical).
17. Diagnosis: PCOS diagnosis outcome (Categorical: Yes/No).

## Repository Structure
```
/pcos_prediction
│
├── /backend
│   ├── pcos_load.py               # Model loading and prediction logic
│   ├── pcos_predict.py            # User input handling and prediction requests
│   ├── Dockerfile                 # Dockerfile for backend
│   └── docker-compose.yml         # Docker Compose configuration
│
├── /frontend
│   ├── /templates
│   │   ├── index.html             # Main HTML file for the application
│
├── /data
│   ├── dv.pkl                       # DictVectorizer file
│   ├── random_forest_model.pkl      # Trained model file
│   └── pcos_prediction_dataset.csv  # Dataset for predictions
│
├── requirements.txt
└── README.md
```
## Prerequisites
- Python 3.x
- Flask
- Docker (for Docker setup)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ajith-Kumar-Nelliparthi/Machine-Learning-Projects.git
   cd PCOS_Prediction
   ```

2. Install the required Python packages:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. Install dependencies using Pipenv:
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
    python pcos_load.py
    ```
   ![Screenshot 2025-01-23 214258](https://github.com/user-attachments/assets/e422426f-bc1d-4779-b213-ac10056bcc28)



3. In another terminal, run the prediction script:
    ```sh
    python pcos_predict.py
    ```
   ![Screenshot 2025-01-23 214319](https://github.com/user-attachments/assets/7c3bedac-9ec5-40d1-ab7e-7933e5a3c22e)

## Docker Setup
1. Build and run the Docker container:
   ```bash
   docker-compose up --build
   ```
   ![Screenshot 2025-01-24 164747](https://github.com/user-attachments/assets/68a5750e-5c8a-44b4-b3eb-7f02cdf7bede)


2. Access the application at `http://localhost:8000/index`.

  ![Screenshot 2025-01-24 161917](https://github.com/user-attachments/assets/2cafc3fa-e137-4807-a784-e156ba5c849d)

## Usage

Once the application is running, you can use it to predict PCOS by providing relevant input data.
Example input features may include hormonal levels, body mass index (BMI), and other medical parameters.
Ensure that the input data is preprocessed similarly to the training data.

## Contributing

Contributions are welcome! If you'd like to contribute, follow these steps:
1. Fork the repository.
2. Create a feature branch (``git checkout -b feature-branch``).
3. Commit your changes (``git commit -m "Add new feature"``).
4 .Push to the branch (``git push origin feature-branch``).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.



