from django.urls import path,include
from .import views

urlpatterns = [
    path('',views.weather_view,name='Weather View'),
   # path('', include('forecast.urls')),
    
]
