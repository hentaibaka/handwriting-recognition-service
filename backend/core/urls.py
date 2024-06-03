from django.urls import path
from .views import *


urlpatterns = [
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('register/', RegisterView.as_view(), name='register'),
    path('profile/', UserProfileView.as_view(), name='profile'),
    path('profile/change-password/', ChangePasswordView.as_view(), name='change-password'),
    path('profile/update-user/', UpdateUserView.as_view(), name='update-user'),
    path('password_reset/', PasswordResetView.as_view(), name='password_reset'),
    path('get-csrf-token/', get_csrf_token, name='get-csrf-token'),
]
