# Predicting Daily Electricity Demand in New York, USA
The project is a machine learning prediction service that predicts the daily electricity demand in megawatthours in New York, USA.

As features, the project uses historical daily demand, daily average temperature (celsius), and whether the date was a US Federal holiday or not.

Training data consists of the years 2017 - 2021. The data for historical daily demand is obtained from [US Energy Information Administration (EIA)][0]  and the data for historical average daily temperature is obtained from [National Oceanic and Atmospheric Administration (NOAA)][1]. The US Federal holidays are obtained from the ``USFederalHolidayCalendar`` library in ``pandas.tseries.holiday``. The temperature data for the daily scheduled batch predictions is obtained from [WeatherAPI][2].

Along with the model's predictions we also provide the forecasted demand from EIA and find that our model's predicted values turn out to be fairly close to those of the official EIA forecast.


[0]: <https://www.eia.gov/>
[1]: <https://www.noaa.gov/>
[2]: <https://www.weatherapi.com/>


## Pipelines
The prediction service is built using separate feature, training and inference pipelines, as described below. Hopsworks is used as feature store and model registry, daily instance generation and batch inference are deployed as functions in Modal and the UIs for online inference and monitoring are implemented using Hugginface spaces. 

## Feature pipeline
Implemented in ``feature_pipeline.ipynb`` as a Jupyter notebook. The features used for training and prediction are daily demand for the NY area (prediction target), daily average temperature for the same area, day and month (represented as an integer feature) and bank holiday status (true or false, represented as a binary feature).

To train the model and initialize the feature store with data, historic demand (from EIA) and weather data (from NOAA) for a span of 5 years (years 2017-2021) is used, in total 1826 instances. The data gathering and preparation steps applied are the following.

Get demand (in MWh) data from EIA API
Get historic weather data (temperature in °C) from NOAA API
Merge dataframes on the date
Add "day", "month", and "holiday" (by comparing the date with the Pandas bank holiday database) features

## Training pipeline
Implemented in ``training.ipynb`` as Jupyter notebook. The model used for the predictions is based on the XGBoost Regressor implemented in the xgboost Python package. The model is trained on the historic demand and weather data prepared in the Feature pipeline.

First the training data, retrieved from the feature store, is split in training (80%) and testing data (20%, used later to estimate the performance of the model). No particular data preparation techniques are used except for imputation (even though we did not find any missing data) and Min-max normalization of the temperature. To avoid leaking information from the test split to the training split and to create a re-usable data preparation pipeline which can be stored together with the model in the model registry, the data preparation is implemented through Scikit-learn pipelines and fitted to the training data only.

To find a reasonable hyperparameter configuration a randomized search with 10-fold cross validation is performed. The best performing model, as measured using the mean absolute error (MAE), is then dumped (including the data preparation pipeline) and uploaded to the model registry. The estimated MAE for the model, observed on the test set, is around 14GWh (around 3-4%) which seems to be in line with EIA's own forecasts.
```
NYC-CURRENT-PRICE-PREDICTION/ 
├── .env 
├── api_key 
├── WEATHER_API_KEY
├── batch-daily.py 
├── feature_pipeline.ipynb 
├── feature-daily.py 
├── hugging-spaces-electricity.py 
├── model/ 
│ └── ny_elec_model.pkl 
├── noaa_weather_lag_2017-2021.csv 
├── training.ipynb
```


## Files and Directories

- `batch-daily.py`: Script to run daily batch predictions.
- `feature_pipeline.ipynb`: Jupyter notebook for feature engineering pipeline.
- `feature-daily.py`: Script to generate daily features for prediction.
- `hugging-spaces-electricity.py`: Script to deploy the model using Gradio for interactive predictions.
- `model/`: Directory containing the trained machine learning model.
- `noaa_weather_lag_2017-2021.csv`: Historical weather data.
- `training.ipynb`: Jupyter notebook for training the machine learning model.
- `.env`: Environment file containing API keys and other secrets.
- `api_key`: File containing the API key for accessing external data sources.
- `WEATHER_API_KEY`: File containing the weather API key.

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/Ajith-Kumar-Nelliparthi/Machine-Learning-Projects.git
    cd Machine-Learning-Projects/NYC-CURRENT-PRICE-PREDICTION
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up the environment variables:
    - Create a [.env](http://_vscodecontentref_/6) file in the root directory and add your API keys:
        ```
        EIA_API_KEY=your_eia_api_key
        WEATHER_API_KEY=your_weather_api_key
        HOPSWORKS_API_KEY=your_hopsworks_api_key
        ```

## Usage

### Training the Model

1. Open the [training.ipynb](http://_vscodecontentref_/7) notebook and run all cells to train the machine learning model.
2. The trained model will be saved in the [model](http://_vscodecontentref_/8) directory as `ny_elec_model.pkl`.

### Running Daily Batch Predictions

1. Run the [batch-daily.py](http://_vscodecontentref_/9) script to generate daily predictions:
    ```sh
    python batch-daily.py
    ```

### Generating Daily Features

1. Run the [feature-daily.py](http://_vscodecontentref_/10) script to generate daily features for prediction:
    ```sh
    python feature-daily.py
    ```

### Deploying the Model with Gradio

1. Run the [hugging-spaces-electricity.py](http://_vscodecontentref_/11) script to deploy the model using Gradio:
    ```sh
    python hugging-spaces-electricity.py
    ```
    ![Screenshot 2025-01-31 163938](https://github.com/user-attachments/assets/15867e3e-00c1-46ba-8be6-12eb72cc4b73)

    ![Screenshot 2025-01-31 163702](https://github.com/user-attachments/assets/3d332278-6e8b-4008-ab3f-940383207a68)


## Data Sources

- Historical weather data from NOAA.
- Electricity demand data from the EIA API.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.



























