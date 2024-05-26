from django.urls import path, include
from .views import *


urlpatterns = [
    path('metrics/', MetricsView.as_view()),
]