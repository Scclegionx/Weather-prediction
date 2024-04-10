from django.urls import path, include
from . import views


urlpatterns = [
    path('', views.home, name='HOME'),
    path('test/', views.test, name='TEST')
]
