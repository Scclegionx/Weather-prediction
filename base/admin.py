from django.contrib import admin
from .models import WeatherData, Prediction

admin.site.register(WeatherData)
admin.site.register(Prediction)