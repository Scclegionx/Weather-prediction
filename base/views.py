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
        
        rf = RandomForestRegressor()
        print(weather.dtypes)

        # Lặp qua 4 lần để tính và thêm dữ liệu cho 4 ngày mới nhất
        for _ in range(4):
            weather = calculate_and_append_3_day_average(weather)

        backtest_result = backtestrf(weather, rf, predictors)

    
        
        # Paginate the weather data
        paginator = Paginator(weather, 10)  
        page_number = request.GET.get('weather_page')
        page_obj = paginator.get_page(page_number)

        # Paginate the backtest_result data
        page_number2 = request.GET.get('backtest_page')
        paginatorbt = Paginator(backtest_result, 10)  
        page_objbt = paginatorbt.get_page(page_number2)

        context = {
            'weather_html': page_obj.object_list.to_html(), 'page_obj': page_obj,
            'backtest_html': page_objbt.object_list.to_html(), 'page_objbt' : page_objbt
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
    # Lấy dữ liệu của 3 ngày trước đó
    previous_days_data = data.iloc[-3:]
    # Tính trung bình số liệu của từng cột trên cùng một hàng của ngày tiếp theo
    average_data = previous_days_data.mean(axis=0)
    # Thêm dữ liệu trung bình vào bảng dữ liệu weather
    data.loc[data.index[-1] + pd.DateOffset(days=1)] = average_data
    return data

def backtestrf(weather, model, predictors, start=365, step=90):
  all_predictions = []
  for i in range(start, weather.shape[0], step):
    train = weather.iloc[:i,:]
    test = weather.iloc[i:(i+step),:]

    model.fit(train[predictors], train["target"])

    preds = model.predict(test[predictors])

    preds = pd.Series(preds, index=test.index)
    combined = pd.concat([test["target"], preds], axis = 1)

    combined.columns = ["actual", "prediction"]
    combined["diff"] = (combined["prediction"] - combined["actual"]).abs()

    all_predictions.append(combined)
  return pd.concat(all_predictions)
