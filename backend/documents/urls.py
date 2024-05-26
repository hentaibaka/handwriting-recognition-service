from django.urls import path
from .views import *


urlpatterns = [
    path('recognize/', RecognizeImage.as_view()),
    path('demodocs/', DemoDocsView.as_view()),
]
