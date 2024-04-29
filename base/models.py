from django.db import models
import os

class WeatherData(models.Model):
    csv_file = models.FileField(upload_to='csv_files/', null=True, blank=True)
    updated = models.DateTimeField(auto_now = True)
    uploaded = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        verbose_name = 'Weather Data'
        verbose_name_plural = 'Weather Data'

    def save(self, *args, **kwargs):
        if self.csv_file:
            filename = os.path.basename(self.csv_file.name)
            self.name = os.path.splitext(filename)[0]  
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

class Prediction(models.Model):
    datetime = models.DateTimeField()
    tempmax = models.FloatField()
    tempmin = models.FloatField()
    humidity = models.FloatField(null=True, blank=True)
    location = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f"Prediction for {self.location} on {self.datetime}"