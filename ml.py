from django.shortcuts import render

# Create your views here.
import requests  # This library helps us to fetch data from API

import pandas as pd  # for handling and analyzing data

from sklearn.model_selection import train_test_split  # to split data into training and testing sets

import numpy as np  # for numerical operations

from sklearn.preprocessing import LabelEncoder  # to convert categorical data into numerical values

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # models for classification and regression

from sklearn.metrics import mean_squared_error  # to measure the accuracy of our predictions

from datetime import datetime, timedelta  # to handle date and time
import pytz

from django.http import HttpResponse

def weather_view(request):
    return HttpResponse('<h1> Weather app </h1>')


API_KEY = '283511028abb4bdc87d8436465614c1d' #replace with you

BASE_URL = 'https://api.openweathermap.org/data/2.5'


#fetching
def get_current_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    
    response = requests.get(url)  # Send the GET request to the API
    data=response.json()
       
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': data['main']['humidity'],
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind'].get('deg', 'N/A'),
        'pressure': data['main']['pressure'],
        'WindGustSpeed': data['wind']['speed']
    }

#read historical data
def read_historical_data(filename):
    df=pd.read_csv(filename)
    df=df.dropna()
    df=df.drop_duplicates()
    return df

#3 prepare data for training
def prepare_data(data):
    le = LabelEncoder()  # Create a LabelEncoder instance

    # Encode categorical variables
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

    # Define feature variables and target variable
    X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 
              'Humidity', 'Pressure', 'Temp']]
    y = data['RainTomorrow']  # Target variable

    return X, y, le  # Return featur

#4 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

def train_rain_model(X, y):

     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     model = RandomForestClassifier(n_estimators=100, random_state=42)
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     

     mse = mean_squared_error(y_test, y_pred)
     print("Mean Squared Error for Rain Model:", mse)
     
     return model
    
    #5 
    
def prepare_regression_data(data, feature):
    X, y = [], []  # initialize lists for feature and target values
    
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])
    
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    
    return X, y

#6
def train_regression_model(x, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x, y)
    return model

#7
def predict_future(model, current_value):
    predictions = [current_value]
    
    for i in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    
    return predictions[1:]

#weather fxn
def weather_view():
    # Get city input from the user
    city = input('Enter any city name: ')

    # Fetch current weather data (you need to define `get_current_weather`)
    current_weather = get_current_weather(city)
    print(f"Current weather in {city}: {current_weather}")

    # Load historical weather data (you need to define `read_historical_data`)
    historical_data = read_historical_data('/content/weather.csv')

    # Prepare and train the rain prediction model (functions need implementation)
    x, y,le = prepare_data(historical_data)  # Prepare data for training
    rain_model = train_rain_model(x, y)  # Train the model
    
    wind_deg= current_weather['wind_gust_dir'] % 360
    
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75), ("N", 348.75, 360)
    ]
compass_direction=next(point for point,start,end in compass_points if start < wind_deg <end)

compass_direction_encoded =le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1


    # Use the model to predict based on current weather data (assuming a format)

current_data = {
    'MinTemp': current_weather.get['temp_min', None],  # Minimum temperature
    'MaxTemp': current_weather.get['temp_max', None],  # Maximum temperature
    'WindGustDir': compass_direction_encoded,          # Encoded wind direction
    'WindGustSpeed': current_weather['wind_speed', None],  # Wind speed
    'Humidity': current_weather['humidity', None],         # Humidity percentage
    'pressure': current_weather['pressure', None],
    'Temp':current_weather['current_temp'],
}
current_df=pd.DataFrame([current_data])

#rain predict
rain_prediction=rain_model.predict(current_df)[0]
X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')

    # Prepare regression data for humidity
X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
# Train regression models
temp_model = train_regression_model(X_temp, y_temp)
hum_model = train_regression_model(X_hum, y_hum)

#future prediction
future_temp=predict_future(temp_model,current_weather['temp_min'])
future_humidity=predict_future(hum_model,current_weather['humidity'])

#prepare time for predition
timezone = pytz.timezone("Asia/Karachi")

    # Get the current time in the specified timezone
now = datetime.now(timezone)
# Calculate the next hour and adjust minutes, seconds, and microseconds
next_hour = now + timedelta(hours=1)
next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
# Generate future times (next 5 hours)
future_times = [(next_hour + timedelta(hours=i)).strftime("%H:%M") for i in range(5)]
# Display results

"""
print(f"City: {city}, {current_weather['country']}")
print(f"Current Temperature: {current_weather['current_temp']}°C")
print(f"Feels Like: {city}, {current_weather['feels_like']}")
print(f"Minimum Temperature: {current_weather['temp_min']}°C")
print(f"Maximum Temperature: {current_weather['temp_max']}°C")
print(f"Humidity: {current_weather['humidity']}°C")
print(f"Weather Prediction: {current_weather['description']}°C")
print(f"Future Times (next 5 hours): {', '.join(future_times)}")
"""