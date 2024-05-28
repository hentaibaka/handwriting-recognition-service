from django.urls import path
from .views import *


urlpatterns = [
    path('recognize/', RecognizeImage.as_view()),
    path('demodocs/<int:page_id>/text-file/', PageTextFileView.as_view(), name='page-text-file'),
    path('demodocs/<int:page_id>/pdf-file/', PagePDFFileView.as_view(), name='page-pdf-file'),
    path('demodocs/', DemoDocsView.as_view()),
]
