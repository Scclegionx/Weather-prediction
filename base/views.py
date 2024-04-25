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
from datetime import datetime, timedelta

def home(request):
    context = {}
    return render(request, 'base/home.html', context)

def test(request):
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
                 'severerisk', 'moonphase', 'target']
        weather[float_columns] = weather[float_columns].astype(float)

        for col in ["tempmax", "tempmin", "precip"]:
            weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
            weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys=False).apply(expand_mean)

        predictors = weather.columns[~weather.columns.isin(["target"])]
        predictors2 = weather.columns[~weather.columns.isin(["target-tempmin"])]
         
        rf = RandomForestRegressor()

        # Lặp qua 4 lần để tính và thêm dữ liệu cho 4 ngày mới nhất
        for _ in range(4):
            weather = calculate_and_append_3_day_average(weather)

        
        backtest_result = backtestrf(weather, rf, predictors)
        backtest_result2 = backtestrf2(weather, rf, predictors2)  # Corrected function call

        tempmin_list = backtest_result2['prediction2'].values.tolist()

        print(backtest_result)


        last_three_days = backtest_result.tail(3)
        last_three_days_tempmin = backtest_result2['prediction2'].tail(3)

# Chuẩn bị dữ liệu cho 3 ngày cuối cùng
        forecast_days_data = [
    {
        'datetime': last_three_days.index[-3],
        'tempmax': last_three_days['prediction'][-3],
        'tempmin': last_three_days_tempmin[-3]
    },
    {
        'datetime': last_three_days.index[-2],
        'tempmax': last_three_days['prediction'][-2],
        'tempmin': last_three_days_tempmin[-2]
    },
    {
        'datetime': last_three_days.index[-1],
        'tempmax': last_three_days['prediction'][-1],
        'tempmin': last_three_days_tempmin[-1]
    }
    
]
        print(forecast_days_data)

        zipped_forecast_data = [
            {
                'datetime': forecast_days_data[i]['datetime'],
                'tempmax': forecast_days_data[i]['tempmax'],
                'tempmin': forecast_days_data[i]['tempmin']
            } for i in range (3)
        ]

        # Paginate the weather data
        paginator = Paginator(weather, 10)  
        page_number = request.GET.get('weather_page')
        page_obj = paginator.get_page(page_number)

        # Paginate the backtest_result data
        page_number2 = request.GET.get('backtest_page')
        paginatorbt = Paginator(backtest_result, 10)  
        page_objbt = paginatorbt.get_page(page_number2)

        page_number3 = request.GET.get('backtest_page2')
        paginatorbt2 = Paginator(backtest_result2, 10)  
        page_objbt2 = paginatorbt2.get_page(page_number3)

        context = {
            'zipped_forecast_data': zipped_forecast_data,
            'weather_html': page_obj.object_list.to_html(), 'page_obj': page_obj,
            'backtest_html': page_objbt.object_list.to_html(), 'page_objbt': page_objbt,
            'forecast_days_data': forecast_days_data,  
            'tempmin_list': tempmin_list,
}

        return render(request, 'base/test.html', context)
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

def backtestrf2(weather, model, predictors2, start=365, step=90):
    # Khoi tao danh sach du doan 2
    all_predictions2 = []
    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i,:]
        test = weather.iloc[i:(i+step),:]
        
        model.fit(train[predictors2], train["target-tempmin"])
        
        preds2 = model.predict(test[predictors2])
        preds2 = pd.Series(preds2, index=test.index)
        combined2 = pd.concat([test["target-tempmin"], preds2], axis=1)
        combined2.columns = ['actual', 'prediction2']
        combined2['actual'] = pd.to_numeric(combined2['actual'], errors='coerce')
        combined2['prediction2'] = pd.to_numeric(combined2['prediction2'], errors='coerce')
        combined2["diff"] = (combined2["prediction2"] - combined2["actual"]).abs()
        
        all_predictions2.append(combined2)
    return pd.concat(all_predictions2)


