from django.urls import path
from .views import generate_data, generate_image


urlpatterns = [
    path('generate/', generate_data),
    path('get_image/', generate_image),
]
