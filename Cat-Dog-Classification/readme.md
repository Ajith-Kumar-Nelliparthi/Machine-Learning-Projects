# Cat-Dog Classification Web Application

## Overview

The Cat-Dog Classification Web Application is a deep learning-based project that classifies images of cats and dogs. Users can upload an image, and the model will predict whether the image contains a cat or a dog.


https://github.com/user-attachments/assets/89967ea7-4d02-4c7f-8f58-f22ae89ae126



## Features
1. Simple and intuitive web interface
2. Image upload functionality
3. Deep learning-based classification
4. Real-time prediction results
5. Lightweight and efficient deployment

## Tech Stack
1. Frontend: HTML, CSS, JavaScript
2. Backend: Flask
3. Model: Convolutional Neural Network (CNN) using TensorFlow/Keras
5. Deployment: Docker

## Dataset 
[Download link](https://www.microsoft.com/en-us/download/details.aspx?id=54765)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Docker
- Python 3.10

### Steps

1. Clone the repository:

    ```sh
   git clone https://github.com/Ajith-Kumar-Nelliparthi/Machine-Learning-Projects.git
   cd Machine-Learning-Projects/Cat-Dog-Classification/web-application
    ```

2. Install dependencies:
    ```sh
   pip install -r requirements.txt
    ```
3. Build the Docker image:

    ```sh
    docker build -t cat-dog-classifier .
    ```

4. Run the Docker container:

    ```sh
    docker run -p 8000:5000 cat-dog-classifier
    ```

5. Open your browser and go to `http://localhost:8000` to access the application.

## Usage

1. Open the web application in your browser.
2. Upload an image of a cat or dog.
3. Click the "Predict" button to get the classification result.


## Project Structure
```
Cat-Dog-Classifier/
├── app.py                         The main Flask application file.
├── Dockerfile                     The Dockerfile to build the Docker image.
├── pet_classifier_model.h5        The pre-trained TensorFlow model.
├── Pipfile                        Files for managing Python dependencies.
├── Pipfile.lock
├── requirements.txt
├── static/
│   └── style.css                  CSS file for styling the web application.
└── templates/
    └── index.html                 HTML template for the web application.

```
## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.



## Screenshots

![Screenshot 2025-02-15 143836](https://github.com/user-attachments/assets/ea968f2c-3d50-488b-879f-bedb7ead4883)


## Future Improvements

→ Enhance model accuracy with data augmentation\
→  Implement multi-class classification for more pet types\
→  Add real-time image capture for predictions


## Contact

For any queries, reach out to:

GitHub: [Ajith-Kumar-Nelliparthi](https://github.com/Ajith-Kumar-Nelliparthi)





