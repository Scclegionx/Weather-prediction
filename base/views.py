from django.shortcuts import render
from io import StringIO
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from .models import WeatherData
from django.http import HttpResponse
import csv
from django.core.paginator import Paginator
from tensorflow import keras
from keras.src.layers import Dense, SimpleRNN, LSTM
from keras.src.models import Sequential
import pandas as pd
import numpy as np

from .models import Prediction
from datetime import datetime, timedelta


CACHE_TIMEOUT = 24 * 60 * 60


def home(request):
    context = {}
    return render(request, 'base/home.html', context)

def test(request):
    # Set today's date to "2024-02-27"
    today = datetime(2024, 3, 2)
    
    # Calculate the end_date as "2024-03-01"
    end_date = datetime(2024, 3, 5) + timedelta(days=1)
    # Get today's date
    #today = datetime.now().date()
    
    # Calculate the date range for the next 4 days
    #end_date = today + timedelta(days=4)
    
    # Check if predictions for the next 4 days exist in the database
    predictions_exist = Prediction.objects.filter(datetime__gte=today, datetime__lt=end_date).exists()
    print(today)
    print(end_date)
    print(predictions_exist)
    if not predictions_exist:
        # If predictions don't exist, generate new predictions
        forecast_days_data = weatherPredict()  # This function should generate the forecast data
        
        # Add new predictions to the database
        for data in forecast_days_data:
            prediction = Prediction.objects.create(
                datetime=data['datetime'],
                tempmax=data['tempmax'],
                tempmin=data['tempmin']
            )
        
        predictions = forecast_days_data
    else:
        # If predictions exist, retrieve them from the database
        predictions = Prediction.objects.filter(datetime__gte=today, datetime__lt=end_date).values()
    
    context = {'dict': predictions}
    return render(request, 'base/test.html', context)


def weatherPredict():
    weather_data = WeatherData.objects.all()
    if weather_data.exists():
        csv_data = []
        for data in weather_data:
            # Read the contents of the CSV file
            csv_file = data.csv_file.read().decode('utf-8')
            # Parse the CSV data
            csv_reader = csv.reader(StringIO(csv_file))
            # Skip the header row
            next(csv_reader)
            for row in csv_reader:
                # Convert empty strings to "NaN"
                row = [None if cell == '' else cell for cell in row]
                csv_data.append(row)
        
        # Process CSV data and perform backtesting
        weather = pd.DataFrame(csv_data, columns=['name', 'datetime', 'tempmax', 'tempmin','temp', 'feelslikemax', 'feelslikemin',
       'feelslike', 'dew', 'humidity', 'precip', 'precipprob', 'precipcover','preciptype',
       'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir',
       'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation',
       'solarenergy', 'uvindex', 'severerisk', 'sunrise', 'sunset',
       'moonphase', 'conditions', 'description', 'icon', 'stations']) 
        
        unwanted_columns = ["name", "sunrise", "sunset", "conditions", "icon", "stations", "description", "preciptype"]
        weather = weather.drop(columns=unwanted_columns, errors='ignore')

        null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]
        valid_columns = weather.columns[null_pct < .05]
        weather = weather[valid_columns].copy()
        weather = weather.ffill()
        weather = weather.bfill()

        weather['datetime'] = pd.to_datetime(weather['datetime'])
        weather.set_index('datetime', inplace=True)

        weather["target"] = weather.shift(-1)["tempmax"]
        weather["target-tempmin"] = weather.shift(-1)["tempmin"]
        weather = weather.ffill()

        weather = weather.iloc[14:,:]
        weather = weather.fillna(0)

        float_columns = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity', 
                 'precip', 'precipprob', 'precipcover', 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir', 
                 'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex', 
                 'severerisk', 'moonphase', 'target', 'target-tempmin']
        weather[float_columns] = weather[float_columns].astype(float)

        for col in ["tempmax", "tempmin", "precip"]:
            weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
            weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys=False).apply(expand_mean)

        predictors = weather.columns[~weather.columns.isin(["target", "target-tempmin"])]
        
        rf = RandomForestRegressor()

        # Lặp qua 4 lần để tính và thêm dữ liệu cho 4 ngày mới nhất
        for _ in range(4):
            weather = calculate_and_append_3_day_average(weather)
        
        #tempmxa
        # Random Forest
        backtest_result = backtestrf(weather, rf, predictors)

        #RNN Simple
        X_rnn = weather[predictors].values.reshape(-1, 1, len(predictors))

        rnn = Sequential()
        rnn.add(SimpleRNN(64, input_shape=(X_rnn.shape[1], X_rnn.shape[2]), activation='relu'))
        rnn.add(Dense(32, activation='relu'))
        rnn.add(Dense(1, activation='linear'))
        rnn.compile(optimizer='adam', loss='mse')

        y_rnn = weather["target"].values
        rnn.fit(X_rnn, y_rnn, epochs=10, batch_size=32, verbose=0)
        preds_rnn = rnn.predict(X_rnn)
        datetime_index = weather.index

        diff = abs(weather['target'] - preds_rnn.flatten())
        # Create a DataFrame for predictions
        predictions_df = pd.DataFrame({'index': datetime_index, 'actual': weather['target'], 'predictions': preds_rnn.flatten(), 'diff': diff})
        predictions_df.set_index('index', inplace=True)
        #print(predictions_df)

        # LSTM

        # Reshape input data for LSTM
        X_lstm = weather[predictors].values.reshape(-1, 1, len(predictors))

        lstm = Sequential()
        lstm.add(LSTM(64, input_shape=(X_lstm.shape[1], X_lstm.shape[2]), activation='relu'))
        lstm.add(Dense(32, activation='relu'))
        lstm.add(Dense(1, activation='linear'))
        lstm.compile(optimizer='adam', loss='mse')

        # lstm.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

        # preds_lstm = lstm.predict(X_test)

        y_rnn = weather["target"].values
        # Fit the model
        lstm.fit(X_lstm, y_rnn, epochs=10, batch_size=32)

        preds_lstm = lstm.predict(X_lstm)

        datetime_index = weather.index

        # Calculate the difference between actual and predicted values
        diff = abs(weather['target'] - preds_lstm.flatten())

        # Create a DataFrame for predictions
        predictions_lstn = pd.DataFrame({'index': datetime_index, 'actual': weather['target'], 'predictions': preds_lstm.flatten(), 'diff': diff})

        # Optionally, set the index to the datetime index
        predictions_lstn.set_index('index', inplace=True)

        # Display the predictions DataFrame
        #print(predictions_lstn)

        # ensemble

        # Get the common indices between backtest_result and predictions_df
        common_indices = backtest_result.index.intersection(predictions_df.index)

        # Filter backtest_result and predictions_df to keep only the common indices
        backtest_result_filtered = backtest_result.loc[common_indices]
        predictions_df_filtered = predictions_df.loc[common_indices]
        predictions_lstn_filtered = predictions_lstn.loc[common_indices]

        # Combine predictions
        combined_preds = np.vstack((backtest_result_filtered['prediction'].values, predictions_df_filtered['predictions'].values,predictions_lstn_filtered['predictions'].values))

        # Take average of predictions
        ensemble_preds = np.mean(combined_preds, axis=0)


        # Create DataFrame for ensemble predictions
        ensemble_df = pd.DataFrame({'index': common_indices,
                                    'prediction_backtest': backtest_result_filtered['prediction'],
                                    'prediction_rnn': predictions_df_filtered['predictions'],
                                    'prediction_lstm': predictions_lstn_filtered['predictions'],
                                    'ensemble_prediction': ensemble_preds,
                                    'actual': backtest_result_filtered['actual'],
                                    'diff': np.abs(backtest_result_filtered['actual'] - ensemble_preds)})

        # Display the ensemble predictions DataFrame
        print(ensemble_df)
        #print(ensemble_df["diff"].round().value_counts())


        # tempmin
        # RF
        backtest_result2 = backtestrf2(weather, rf, predictors)
        #print(backtest_result2)

        #RNN

        X_rnn = weather[predictors].values.reshape(-1, 1, len(predictors))

        rnn = Sequential()
        rnn.add(SimpleRNN(64, input_shape=(X_rnn.shape[1], X_rnn.shape[2]), activation='relu'))
        rnn.add(Dense(32, activation='relu'))
        rnn.add(Dense(1, activation='linear'))
        rnn.compile(optimizer='adam', loss='mse')

        y_rnn = weather["target-tempmin"].values
        rnn.fit(X_rnn, y_rnn, epochs=10, batch_size=32, verbose=0)
        preds_rnn2 = rnn.predict(X_rnn)

        datetime_index = weather.index

        # Calculate the difference between actual and predicted values
        diff = abs(weather['target-tempmin'] - preds_rnn2.flatten())

        # Create a DataFrame for predictions
        predictions_df2 = pd.DataFrame({'index': datetime_index, 'actual': weather['target-tempmin'], 'predictions2': preds_rnn2.flatten(), 'diff': diff})

        # Optionally, set the index to the datetime index
        predictions_df2.set_index('index', inplace=True)

        # Display the predictions DataFrame
        #print(predictions_df2)


        #LSTM

        X_lstm = weather[predictors].values.reshape(-1, 1, len(predictors))

        # # Normalize the data
        # X_lstm = (X_lstm - X_lstm.mean()) / X_lstm.std()

        # # Split the data into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(X_lstm, weather["target"].values, test_size=0.2, random_state=42)

        lstm2 = Sequential()
        lstm2.add(LSTM(64, input_shape=(X_lstm.shape[1], X_lstm.shape[2]), activation='relu'))
        lstm2.add(Dense(32, activation='relu'))
        lstm2.add(Dense(1, activation='linear'))
        lstm2.compile(optimizer='adam', loss='mse')

        # lstm.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

        # preds_lstm = lstm.predict(X_test)

        y_rnn = weather["target-tempmin"].values
        # Fit the model
        lstm2.fit(X_lstm, y_rnn, epochs=10, batch_size=32)

        preds_lstm2 = lstm2.predict(X_lstm)

        
        datetime_index = weather.index

        # Calculate the difference between actual and predicted values
        diff = abs(weather['target-tempmin'] - preds_lstm2.flatten())

        # Create a DataFrame for predictions
        predictions_lstn2 = pd.DataFrame({'index': datetime_index, 'actual': weather['target-tempmin'], 'predictions2': preds_lstm2.flatten(), 'diff': diff})

        # Optionally, set the index to the datetime index
        predictions_lstn2.set_index('index', inplace=True)

        # Display the predictions DataFrame
        #print(predictions_lstn2)

        # ensemble

        # Get the common indices between backtest_result and predictions_df
        common_indices = backtest_result2.index.intersection(predictions_df.index)

        # Filter backtest_result and predictions_df to keep only the common indices
        backtest_result_filtered = backtest_result2.loc[common_indices]
        predictions_df_filtered = predictions_df2.loc[common_indices]
        predictions_lstn_filtered = predictions_lstn2.loc[common_indices]

        # Combine predictions
        combined_preds = np.vstack((backtest_result_filtered['predictors2'].values, predictions_df_filtered['predictions2'].values,predictions_lstn_filtered['predictions2'].values))

        # Take average of predictions
        ensemble_preds = np.mean(combined_preds, axis=0)

        # Create DataFrame for ensemble predictions
        ensemble_df2 = pd.DataFrame({'index': common_indices,
                                    'prediction_backtest': backtest_result_filtered['predictors2'],
                                    'prediction_rnn': predictions_df_filtered['predictions2'],
                                    'prediction_lstm': predictions_lstn_filtered['predictions2'],
                                    'ensemble_prediction': ensemble_preds,
                                    'actual': backtest_result_filtered['actual'],
                                    'diff': np.abs(backtest_result_filtered['actual'] - ensemble_preds)})

        # Display the ensemble predictions DataFrame
        print(ensemble_df2)
        #print(ensemble_df2["diff"].round().value_counts())

        last_4_days_tempmax = ensemble_df.tail(4)
        last_4_days_tempmin = ensemble_df2.tail(4)

        forecast_days_data = [
            {
                'datetime': last_4_days_tempmax.index[-4],
                'tempmax': last_4_days_tempmax['ensemble_prediction'][-4],
                'tempmin': last_4_days_tempmin['ensemble_prediction'][-4]
            },
            {
                'datetime': last_4_days_tempmax.index[-3],
                'tempmax': last_4_days_tempmax['ensemble_prediction'][-3],
                'tempmin': last_4_days_tempmin['ensemble_prediction'][-3]
            },
            {
                'datetime': last_4_days_tempmax.index[-2],
                'tempmax': last_4_days_tempmax['ensemble_prediction'][-2],
                'tempmin': last_4_days_tempmin['ensemble_prediction'][-2]
            },
            {
                'datetime': last_4_days_tempmax.index[-1],
                'tempmax': last_4_days_tempmax['ensemble_prediction'][-1],
                'tempmin': last_4_days_tempmin['ensemble_prediction'][-1]
            }
        ]

        zipped_forecast_data = [
            {
                'datetime': forecast_days_data[i]['datetime'],
                'tempmax': forecast_days_data[i]['tempmax'],
                'tempmin': forecast_days_data[i]['tempmin']
            } for i in range (4)
        ]

        return zipped_forecast_data
    else:
        return HttpResponse("Please upload a CSV file.")
        

def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}"
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = weather[label].pct_change()
    return weather

def expand_mean(df):
  return df.expanding(1).mean()


def calculate_and_append_3_day_average(data):
    previous_days_data = data.iloc[-3:]
    
    previous_days_data = previous_days_data.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    previous_days_data = previous_days_data.dropna()
    average_data = previous_days_data.mean(axis=0)

    next_day_index = data.index[-1] + pd.DateOffset(days=1)
    data.loc[next_day_index] = average_data
    
    return data

def backtestrf(weather, model, predictors, start=365, step=90):
    # Khoi tao danh sach du doan
    all_predictions = []
    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i,:]
        test = weather.iloc[i:(i+step),:]
        
        model.fit(train[predictors], train["target"])
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ['actual', 'prediction']
        combined['actual'] = pd.to_numeric(combined['actual'], errors='coerce')
        combined['prediction'] = pd.to_numeric(combined['prediction'], errors='coerce')
        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)


def backtestrf2(weather, model, predictors, start=365, step=90):
  all_predictions = []
  for i in range(start, weather.shape[0], step):
    train = weather.iloc[:i,:]
    test = weather.iloc[i:(i+step),:]

    model.fit(train[predictors], train["target-tempmin"])

    preds = model.predict(test[predictors])

    preds = pd.Series(preds, index=test.index)
    combined = pd.concat([test["target-tempmin"], preds], axis = 1)

    combined.columns = ["actual", "predictors2"]
    combined["diff"] = (combined["predictors2"] - combined["actual"]).abs()

    all_predictions.append(combined)
  return pd.concat(all_predictions)

