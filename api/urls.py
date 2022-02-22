from django.urls import path
from .views import generate_data


urlpatterns = [
    path('generate/', generate_data),
]
