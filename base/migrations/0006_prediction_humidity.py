# Generated by Django 5.0.3 on 2024-04-29 06:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0005_prediction_location'),
    ]

    operations = [
        migrations.AddField(
            model_name='prediction',
            name='humidity',
            field=models.FloatField(blank=True, null=True),
        ),
    ]