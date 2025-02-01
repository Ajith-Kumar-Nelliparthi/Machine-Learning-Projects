import modal
import os
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import datetime
import requests
import hopsworks

LOCAL = False

def feature_elec():
    # Get date two days ago (Demand and demand forecast are 2 days behind)
    prediction_date = datetime.datetime.today() - datetime.timedelta(days=2)
    prediction_date = prediction_date.date()
    print("Date: {}".format(prediction_date))

    # Request for electricity data from EIA API
    url = ('https://api.eia.gov/v2/electricity/rto/daily-region-data/data/'
           '?frequency=daily'
           '&data[0]=value'
           '&facets[respondent][]=NY'
           '&facets[timezone][]=Eastern'
           '&facets[type][]=D'
           '&facets[type][]=DF'
           '&sort[0][column]=period'
           '&sort[0][direction]=desc'
           '&offset=0'
           '&length=5000')
    url = url + '&start={}&end={}&api_key={}'.format(prediction_date, prediction_date, os.environ.get('EIA_API_KEY'))
    data = requests.get(url).json()['response']['data']

    data_demand = data[1]
    data_demand = pd.DataFrame(data_demand, index=[0])

    # Clean DF to same format as fg
    data_demand = data_demand[['period', 'value']].rename(columns={'period': 'date', 'value': 'demand'})
    data_demand['date'] = pd.to_datetime(data_demand['date'], infer_datetime_format=True)
    print("EIA Demand data: \n{}".format(data_demand.head()))

    # Get temperature for date
    weather_api_key = os.environ.get('WEATHER_API_KEY')
    weather_url = ('http://api.weatherapi.com/v1/history.json'
                   '?key={}'
                   '&q=New%20York,%20USA'
                   '&dt={}').format(weather_api_key, prediction_date)

    weather_data = requests.get(weather_url).json()['forecast']['forecastday'][0]['day']['avgtemp_c']
    weather_df = pd.DataFrame({'date': [prediction_date], 'temperature': [weather_data]})
    weather_df['date'] = pd.to_datetime(weather_df['date'], infer_datetime_format=True)
    print("Weather data: \n{}".format(weather_df))

    # Add month/day features
    merged_df = pd.merge(weather_df, data_demand, how='inner', on='date')
    merged_df['day'] = merged_df['date'].dt.dayofweek
    merged_df['month'] = merged_df['date'].dt.month

    # Get bank holidays
    holidays = calendar().holidays(start=merged_df['date'].min(), end=merged_df['date'].max())
    merged_df['holiday'] = merged_df['date'].isin(holidays).astype('int32')
    print("Combined data: \n{}".format(merged_df.head()))

    # Save to Feature Group in Hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()

    fg = fs.get_feature_group(name="ny_elec", version=1)
    fg.insert(merged_df, write_options={"wait_for_job": False})


# Use modal.App instead of modal.Stub
if not LOCAL:
    app = modal.App("nyc-price-prediction")

    # Create Modal Image and install dependencies
    image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install([
        "hopsworks==3.0.4", "seaborn", "joblib", "scikit-learn==1.0.2", "xgboost==1.5", "dataframe-image", "pandas",
        "datetime", "requests", "python-dotenv","numpy==1.26.4","pyarrow"])

    # Define the secret environment variables for EIA API and Weather API keys
    eia_secret = modal.Secret.from_name("EIA_API_KEY")
    weather_secret = modal.Secret.from_name("WEATHER_API_KEY")

    # Define the function to run with Modal
    @app.function(image=image, schedule=modal.Period(days=1),secrets=[
        modal.Secret.from_name("HOPSWORKS_API_KEY")
    ])
    def modal_feature_elec():
        # Set environment variables for API keys
        os.environ['EIA_API_KEY'] = eia_secret
        os.environ['WEATHER_API_KEY'] = weather_secret

        feature_elec()


if __name__ == "__main__":
    if LOCAL:
        feature_elec()
    else:
        app.run()
        # with app.run():
        #    modal_feature_elec()
