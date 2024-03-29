import csv
from io import StringIO
from django.shortcuts import render
from .models import WeatherData

def home(request):
    context = {}
    return render(request, 'base/home.html', context)


def test(request):
    weather_data = WeatherData.objects.all()
    csv_data = []
    for data in weather_data:

        csv_file = data.csv_file.read().decode('utf-8')

        csv_reader = csv.reader(StringIO(csv_file))

        next(csv_reader)
        for row in csv_reader:
            csv_data.append(row)
    context = {'csv_data': csv_data}
    return render(request, 'base/test.html', context)