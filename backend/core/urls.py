from django.urls import path
from .views import *


urlpatterns = [
    path('login/', LoginView.as_view(), name='login'),
    path('register/', RegisterView.as_view(), name='register'),
    path('profile/', UserProfileView.as_view(), name='profile'),
    path('profile/change-password/', ChangePasswordView.as_view(), name='change-password'),
    path('profile/update-user/', UpdateUserView.as_view(), name='update-user'),
    path('get-csrf-token/', get_csrf_token, name='get-csrf-token'),
]
