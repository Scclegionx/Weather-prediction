from django.db import models

class WeatherData(models.Model):
    csv_file = models.FileField(upload_to='csv_files/', null=True, blank=True)
    updated = models.DateTimeField(auto_now = True)
    uploaded = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Weather Data'
        verbose_name_plural = 'Weather Data'

    def __str__(self):
        return f'Weather Data - Uploaded at: {self.uploaded}'
